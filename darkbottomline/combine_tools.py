"""
Higgs Combine integration tools for DarkBottomLine framework.
"""

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uproot
import yaml

from .combine_inputs import (
    ZERO_FLOOR,
    load_region_bin_edges,
    load_region_histogram,
    load_region_syst_histogram,
    region_dir_from_role,
    systematic_applies_to_region,
)


class CombineDatacardWriter:
    """
    Generates Combine-compatible datacards and workspaces.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize datacard writer.

        Args:
            config: Combine configuration dictionary (full combine.yaml)
        """
        self.config = config
        self.datacard_config = config["datacard"]
        self.output_config = config["output"]
        self.advanced_config = config["advanced"]
        self.regions_config = config["regions_config"]

    def _resolve_filename(self, template_key: str, category: str, region_role: str,
                           year: str, mass_point: Optional[str]) -> str:
        """Format an output.{template_key} filename template against this
        datacard's identity (year/category/region/mass_point).

        CR bins have no mass point (background-only) — the {mass_point} token
        is dropped entirely rather than falling back to region_role, which
        would duplicate {region} in the same filename (e.g.
        datacard_2024_1b_CR_Wmunu_CR_Wmunu.txt)."""
        template = self.output_config[template_key]
        if mass_point is None:
            name = template.format(year=year, category=category, region=region_role, mass_point="")
            name = name.replace("__", "_").replace("_.", ".")
            return name
        return template.format(year=year, category=category, region=region_role, mass_point=mass_point)

    def write_datacard(self, region_root_dir: str, output_dir: str,
                        category: str, region_role: str, variable: str,
                        mass_point: Optional[str] = None,
                        blind: bool = True, year: str = "2024",
                        region_dir_override: Optional[str] = None,
                        filename_region: Optional[str] = None) -> str:
        """
        Write a Combine datacard for one (category, region) bin.

        Args:
            region_root_dir: Directory containing hist_*.root files for this era
            output_dir: Output directory for the datacard
            category: "1b" or "2b"
            region_role: region key without category prefix, e.g. "SR", "CR_Wmunu"
            variable: fit/discriminant variable name
            mass_point: signal mass-point label, required for SR bins
            blind: datacard's `observation` line is always -1 (shape-derived,
                matching Run2's convention) regardless of this flag — blind
                instead controls what write_shapes() puts in the shapes.root
                "data_obs" histogram (TotalBkg/Asimov if True, else real data)
            year: data-taking year, used for the lumi_{year} nuisance name
            region_dir_override: histogram-lookup region_dir to use instead of
                region_dir_from_role(region_role) — e.g. "Wlnu" when
                combine_emu merges CR_Wmunu+CR_Wenu into one channel.
                region_role/region_key (used for gated_by_cut and rate_parameter
                matching against regions.yaml/combine.yaml, which only know the
                real per-channel role) are unaffected by this override; only
                the histogram file lookup and resulting bin name change.
            filename_region: label used for the OUTPUT FILENAME only (defaults
                to region_role) — e.g. "Wlnu" so a merged CR's datacard file is
                named/discoverable as "Wlnu", not "CR_Wmunu" (its arbitrarily
                chosen first-encountered per-channel role). Independent of
                region_dir_override, which governs histogram lookup and the
                in-datacard bin name, not the filename.

        Returns:
            Path to generated datacard file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        datacard_file = output_path / self._resolve_filename(
            "datacard_file", category, filename_region or region_role, year, mass_point)

        region_dir = region_dir_override or region_dir_from_role(region_role)
        shapes_filename = self._resolve_filename(
            "shapes_file", category, filename_region or region_role, year, mass_point)

        binning_mode = self.datacard_config.get("binning_mode", "unbinned")
        if binning_mode == "binned":
            datacard_content = self._generate_binned_datacard_content(
                region_root_dir, category, region_role, region_dir, variable,
                mass_point, blind, year, shapes_filename,
            )
        else:
            datacard_content = self._generate_unbinned_datacard_content(
                region_root_dir, category, region_role, region_dir, variable,
                mass_point, blind, year, shapes_filename,
            )

        with open(datacard_file, "w") as f:
            f.write(datacard_content)

        logging.info(f"Datacard written to {datacard_file}")

        return str(datacard_file)

    def _process_keys_for_region(self, is_sr: bool) -> List[str]:
        processes = self.datacard_config["processes"]
        process_keys = [p for p in processes.keys()
                        if not processes[p].get("is_signal", False) and processes[p].get("enabled", True)]
        if is_sr:
            process_keys = ["signal"] + process_keys
        return process_keys

    def _systematics_header_count(self, region_key: str) -> int:
        """kmax counts only lnN/shape systematics — rateParam lines are tracked
        separately by Combine's own parser (HiggsAnalysis/CombinedLimit's
        DatacardParser.py puts them in ret.rateParams, not ret.systs) and must
        NOT be added here. Including them inflates kmax by one per applicable
        rateParam, which combineCards.py/combine reject outright
        ("Found N systematics, expected N+1") — confirmed by direct testing
        against every CR datacard, which all have exactly one rateParam."""
        return len(self.datacard_config["systematics"])

    def _write_bin_block(self, lines: List[str], bin_names: List[str], process_keys: List[str],
                          shapes_filename: str, obs_by_bin: Dict[str, List[float]],
                          rate_by_bin: Dict[str, List[float]]) -> None:
        """Write the shapes/bin/observation/process/rate block shared by both
        binning modes — differs only in how many bin_names are passed (1 for
        unbinned, N for binned) and what obs/rate values are supplied per bin."""
        processes = self.datacard_config["processes"]

        for bin_name in bin_names:
            lines.append(f"shapes * {bin_name} {shapes_filename} $PROCESS $PROCESS_$SYSTEMATIC")
        lines.append("")

        lines.append("# Bin names")
        lines.append("bin".ljust(20) + " ".join(bin_names))
        lines.append("")

        lines.append("# Observation")
        obs_line = "observation".ljust(20)
        for bin_name in bin_names:
            obs_line += " " + " ".join(
                str(v) if v < 0 else str(int(v)) for v in obs_by_bin[bin_name]
            )
        lines.append(obs_line)
        lines.append("")

        lines.append("# Process names")
        bin_line = "bin".ljust(20)
        for bin_name in bin_names:
            bin_line += (" " + bin_name) * len(process_keys)
        lines.append(bin_line)

        idx_line = "process".ljust(20)
        name_line = "process".ljust(20)
        for _ in bin_names:
            signal_index = 0
            bkg_index = 1
            for proc in process_keys:
                idx = signal_index if processes[proc].get("is_signal", False) else bkg_index
                idx_line += f" {idx}"
                name_line += f" {processes[proc]['name']}"
                if not processes[proc].get("is_signal", False):
                    bkg_index += 1
        lines.append(idx_line)
        lines.append(name_line)

        lines.append("# Rates")
        rate_line = "rate".ljust(20)
        for bin_name in bin_names:
            rate_line += " " + " ".join(f"{r:.6f}" for r in rate_by_bin[bin_name])
        lines.append(rate_line)
        lines.append("")

    def _process_hist_key(self, process: str, mass_point: Optional[str]) -> str:
        """Same hist_key convention write_shapes() uses to read a process's
        histogram from the region ROOT file (signal: sig_{mass_point},
        background: plot_group_label)."""
        proc_config = self.datacard_config["processes"][process]
        if proc_config.get("is_signal", False):
            return f"sig_{mass_point}"
        return proc_config["plot_group_label"]

    def _shape_variant_exists(self, region_root_dir: str, category: str, region_dir: str,
                               variable: str, process: str, mass_point: Optional[str],
                               syst_suffix: str) -> bool:
        """Check whether BOTH Up and Down shape-variant files actually contain
        this process's histogram key — mirrors exactly what write_shapes()
        succeeds/skips on, so the datacard never claims a shape systematic
        (writes 1.0) for a process/region where no such shape was written."""
        hist_key = self._process_hist_key(process, mass_point)
        for direction in ("UP", "DOWN"):
            try:
                load_region_syst_histogram(region_root_dir, category, region_dir, variable,
                                            syst_suffix, direction, hist_key)
            except (FileNotFoundError, KeyError):
                return False
        return True

    def _write_systematics_block(self, lines: List[str], bin_names: List[str],
                                  process_keys: List[str], region_key_by_bin: Dict[str, str],
                                  year: str, region_root_dir: str, category: str,
                                  region_dir: str, variable: str,
                                  mass_point: Optional[str]) -> None:
        systematics = self.datacard_config["systematics"]
        lines.append("# Systematic uncertainties")
        for sys_name, sys_config in systematics.items():
            sys_type = sys_config.get("type", "lnN")
            sys_processes = sys_config.get("processes", [])
            gated_by_cut = sys_config.get("gated_by_cut")

            row_name = self._resolve_systematic_name(sys_name, sys_config, year)
            sys_value = self._resolve_systematic_value(sys_name, sys_config, year)

            row_label = f"{row_name} {'lnN' if sys_type == 'lnN' else 'shape'}"
            sys_line = row_label.ljust(20)
            for bin_name in bin_names:
                applies_here = systematic_applies_to_region(
                    self.regions_config, region_key_by_bin[bin_name], gated_by_cut,
                )
                for proc in process_keys:
                    if not (applies_here and proc in sys_processes):
                        sys_line += " -"
                        continue
                    if sys_type == "lnN":
                        sys_line += f" {sys_value:.3f}"
                        continue
                    # shape: only claim it if write_shapes() actually wrote both
                    # Up/Down variants for this process (Section 3's gated_by_cut
                    # controls REGION applicability; this controls whether the
                    # underlying histogram file/key exists at all).
                    exists = self._shape_variant_exists(
                        region_root_dir, category, region_dir, variable, proc,
                        mass_point, sys_config["syst_suffix"],
                    )
                    sys_line += " 1.0" if exists else " -"
            lines.append(sys_line)
        lines.append("")

    def _write_rate_params_block(self, lines: List[str], bin_name: str, region_key: str,
                                  bin_index: Optional[int] = None,
                                  category: Optional[str] = None, is_sr: bool = False) -> None:
        """bin_index (1-based), when given, makes each bin float its process
        normalization INDEPENDENTLY — rate_name gets a _binN suffix, matching
        Run2's real binned convention exactly (verified against the actual
        combined-Run2 datacard: ratewjets_1b_2016_bin1, _bin2, _bin3, _bin4
        are 4 DISTINCT rateParams, not one name repeated across bins with
        different targets). Unbinned mode (bin_index=None) keeps the flat,
        unsuffixed name — a region has only one bin there, so there's
        nothing to disambiguate.

        CR bins (is_sr=False) use combine.yaml's plain rate_parameters block,
        matched by region_key — EXCEPT when combine_emu: true and this CR's
        region_role is the first-encountered half of an EMU_PAIRS-merged pair
        (e.g. CR_Wmunu, standing in for the merged Wlnu channel per
        make_datacard's emu_written_dirs dedup): the per-channel rate_name
        (wjets_1b_Wmunu) is then substituted for the merged one
        (wjets_1b_Wlnu, from sr_rate_parameters[category]'s "merged" field)
        so the CR file's own rateParam matches what SR ties to — reproduced
        directly: without this substitution the merged Wlnu CR file carried
        BOTH wjets_1b_Wlnu (SR's tie) and wjets_1b_Wmunu (the stale
        per-channel entry, still region-key-matched) simultaneously.

        SR (is_sr=True) uses a SEPARATE mechanism (sr_rate_parameters) rather
        than also being listed in rate_parameters[*].regions, because the
        correct SR tie-in set depends on combine_emu's merge state — reproduced
        directly: listing "1b:SR" in both wjets_1b_Wmunu's and wjets_1b_Wenu's
        regions made SR carry BOTH rateParams even when combine_emu: true had
        already merged them into one Wlnu CR, double-applying the correction.
        User-confirmed behavior: unmerged -> SR gets BOTH per-channel
        rateParams (each independently scales SR's process, matching Run2's
        real per-bin CR<->SR tie, verified against ratewjets_1b_2016_bin1
        appearing on both d2016_cat_1b_sr_bin1 and d2016_cat_1b_wl_bin1);
        merged -> SR gets ONE merged rateParam tied to the single merged CR."""
        rate_parameters = self.datacard_config.get("rate_parameters", {})
        sr_config = self.datacard_config.get("sr_rate_parameters", {}).get(category, [])
        combine_emu = self.config.get("combine_emu", True)

        if is_sr:
            rate_names = []
            for entry in sr_config:
                if combine_emu:
                    rate_names.append(entry["merged"])
                else:
                    rate_names.extend(entry["unmerged"])
            applicable = {name: rate_parameters[name] for name in rate_names if name in rate_parameters}
        else:
            raw_applicable = {name: cfg for name, cfg in rate_parameters.items()
                               if region_key in cfg.get("regions", [])}
            if combine_emu:
                # Replace any per-channel name that's part of a merged pair
                # with the merged name, deduplicating so the merged CR file
                # gets exactly one rateParam per process, not one per
                # original channel.
                unmerged_to_merged = {
                    unmerged_name: entry["merged"]
                    for entry in sr_config for unmerged_name in entry["unmerged"]
                }
                applicable = {}
                for name, cfg in raw_applicable.items():
                    merged_name = unmerged_to_merged.get(name)
                    if merged_name is not None:
                        applicable.setdefault(merged_name, rate_parameters.get(merged_name, cfg))
                    else:
                        applicable[name] = cfg
            else:
                applicable = raw_applicable

        if not applicable:
            return
        lines.append("# Rate parameters")
        for rate_name, rate_config in applicable.items():
            rate_processes = rate_config.get("processes", [])
            rate_value = rate_config.get("value", 1.0)
            rate_range = rate_config.get("range", [0.05, 1.95])
            full_rate_name = f"{rate_name}_bin{bin_index}" if bin_index is not None else rate_name
            lines.append(
                f"{full_rate_name} rateParam {bin_name} {','.join(rate_processes)} "
                f"{rate_value:.3f} [{rate_range[0]:.3f},{rate_range[1]:.3f}]"
            )

    def _generate_unbinned_datacard_content(self, region_root_dir: str, category: str,
                                             region_role: str, region_dir: str, variable: str,
                                             mass_point: Optional[str], blind: bool, year: str,
                                             shapes_filename: str) -> str:
        """One Combine channel per region; the shape histogram carries all
        bins internally (imax 1 — a single channel, not one per Recoil bin)."""
        region_key = f"{category}:{region_role}"
        is_sr = region_role == "SR"
        process_keys = self._process_keys_for_region(is_sr)
        # Combine's DatacardParser rejects bin names starting with a digit
        # (category values are "1b"/"2b") — region_dir first, category last,
        # matching Run2's convention of keeping category out of the bin name
        # entirely (it only appears as combineCards.py's cat_1b=/cat_2b= label
        # prefix at merge time). Kept here too for uniqueness across categories
        # within a single un-merged datacard.
        bin_name = f"{region_dir}_{category}"

        n_nuisance = self._systematics_header_count(region_key)

        # imax 1 -> exactly ONE observation value per channel, not one per
        # histogram bin (that was a real bug: a 4-bin shape with imax 1 but a
        # 4-valued observation line is malformed and rejected by Combine).
        # -1 means "shape-derived" — the per-bin breakdown lives in the TH1D
        # written by write_shapes(), matching Run2's own single-bin-channel
        # convention (observation -1) rather than inventing a summed number.
        rate_values = [
            self._get_process_rate(region_root_dir, category, region_dir, variable, proc, mass_point)
            for proc in process_keys
        ]

        lines = []
        lines.append(f"# Datacard for DarkBottomLine {region_key} analysis ({year}) — unbinned")
        lines.append("# Generated automatically by DarkBottomLine framework")
        lines.append("")
        lines.append("imax 1 number of bins")
        lines.append(f"jmax {len(process_keys) - 1} number of processes minus 1")
        lines.append(f"kmax {n_nuisance} number of nuisance parameters")
        lines.append("")

        self._write_bin_block(
            lines, [bin_name], process_keys, shapes_filename,
            obs_by_bin={bin_name: [-1.0]},
            rate_by_bin={bin_name: rate_values},
        )
        self._write_systematics_block(lines, [bin_name], process_keys,
                                       {bin_name: region_key}, year,
                                       region_root_dir, category, region_dir, variable, mass_point)
        self._write_rate_params_block(lines, bin_name, region_key, category=category, is_sr=is_sr)

        return "\n".join(lines)

    def _generate_binned_datacard_content(self, region_root_dir: str, category: str,
                                           region_role: str, region_dir: str, variable: str,
                                           mass_point: Optional[str], blind: bool, year: str,
                                           shapes_filename: str) -> str:
        """Channel-per-bin: each histogram bin of the region becomes its own
        single-bin Combine channel (region_dir_bin1, region_dir_bin2, ...),
        rate=-1 (shape-derived), matching Run2's actual production convention
        (verified against bbDMlimitmodelrateParam_oneRP/datacards/*/*.txt)."""
        region_key = f"{category}:{region_role}"
        is_sr = region_role == "SR"
        process_keys = self._process_keys_for_region(is_sr)

        n_bins = self._get_number_of_bins(region_root_dir, category, region_dir, variable)
        bin_names = [f"{region_dir}_bin{i + 1}" for i in range(n_bins)]

        n_nuisance = self._systematics_header_count(region_key)

        lines = []
        lines.append(f"# Datacard for DarkBottomLine {region_key} analysis ({year}) — binned")
        lines.append("# Generated automatically by DarkBottomLine framework")
        lines.append("")
        lines.append(f"imax {n_bins} number of bins")
        lines.append(f"jmax {len(process_keys) - 1} number of processes minus 1")
        lines.append(f"kmax {n_nuisance} number of nuisance parameters")
        lines.append("")

        # observation=-1 AND rate=-1 (both shape-derived) for every channel,
        # unconditionally — matches Run2's actual convention exactly (checked:
        # every temp_cards/*.txt uses "observation -1" regardless of whether
        # the region is a blinded SR or an unblinded CR). The real-vs-Asimov
        # distinction lives entirely in shapes.root's data_obs content
        # (write_shapes(blind=...)), not in the datacard's observation line.
        obs_by_bin = {bin_name: [-1.0] for bin_name in bin_names}
        rate_by_bin = {bin_name: [-1.0] * len(process_keys) for bin_name in bin_names}

        self._write_bin_block(lines, bin_names, process_keys, shapes_filename,
                               obs_by_bin=obs_by_bin, rate_by_bin=rate_by_bin)
        region_key_by_bin = {bin_name: region_key for bin_name in bin_names}
        self._write_systematics_block(lines, bin_names, process_keys, region_key_by_bin, year,
                                       region_root_dir, category, region_dir, variable, mass_point)
        for i, bin_name in enumerate(bin_names):
            self._write_rate_params_block(lines, bin_name, region_key, bin_index=i + 1,
                                           category=category, is_sr=is_sr)

        return "\n".join(lines)

    @staticmethod
    def _resolve_systematic_name(sys_name: str, sys_config: Dict[str, Any], year: str) -> str:
        """Resolve a systematic's Combine-facing name (datacard row label AND
        shape histogram key suffix — both MUST agree, or text2workspace.py
        can't find "$PROCESS_$SYSTEMATIC"). name_template (e.g.
        "CMS{year}_{sys}") makes a systematic uncorrelated across eras,
        matching Run2's real per-year naming for detector-year-specific SFs
        (CMS2016_eff_b, CMS2017_PU, ...) — correlated/flat systematics
        (JES, JER, pdf, scale, ...) simply omit name_template and keep their
        plain key. This is entirely yaml-driven (combine.yaml's
        datacard.systematics[*].name_template), not hardcoded per systematic
        here."""
        if "name_template" in sys_config:
            return sys_config["name_template"].format(sys=sys_name, year=year)
        return sys_name

    def _resolve_systematic_value(self, sys_name: str, sys_config: Dict[str, Any],
                                   year: str) -> float:
        """Resolve a systematic's numeric value, using the era's year_config if
        value_source points there, else falling back to the flat `value` key."""
        value_source = sys_config.get("value_source")
        if value_source and value_source.startswith("eras[].year_config:"):
            key = value_source.split(":", 1)[1]
            for era in self.config["eras"]:
                if str(era["year"]) == str(year):
                    try:
                        with open(era["year_config"]) as f:
                            year_cfg = yaml.safe_load(f)
                        if key in year_cfg:
                            return float(year_cfg[key])
                    except FileNotFoundError:
                        pass
                    break
        return float(sys_config.get("value", 1.0))

    def _get_number_of_bins(self, region_root_dir: str, category: str,
                             region_dir: str, variable: str) -> int:
        """Read the actual number of bins from the region's TotalBkg histogram."""
        edges = load_region_bin_edges(region_root_dir, category, region_dir, variable)
        return len(edges) - 1

    def _get_observation_values(self, region_root_dir: str, category: str,
                                 region_dir: str, variable: str, blind: bool) -> List[float]:
        """Return observation bin values: TotalBkg (Asimov) if blind, else data_obs."""
        if blind:
            return list(load_region_histogram(region_root_dir, category, region_dir,
                                               variable, "TotalBkg"))
        try:
            return list(load_region_histogram(region_root_dir, category, region_dir,
                                               variable, "data_obs"))
        except KeyError as exc:
            raise KeyError(
                f"Unblind requested for {category}:{region_dir} but 'data_obs' is "
                f"absent from the region ROOT file — histo-production must be rerun "
                f"with --show-data for this region."
            ) from exc

    def _get_process_rate(self, region_root_dir: str, category: str, region_dir: str,
                           variable: str, process: str, mass_point: Optional[str]) -> float:
        """Sum a process's histogram bin values to get its total rate."""
        if self.datacard_config["processes"][process].get("is_signal", False) and mass_point is None:
            raise ValueError(f"mass_point is required to get the signal rate for '{process}'")
        key = self._process_hist_key(process, mass_point)
        values = load_region_histogram(region_root_dir, category, region_dir, variable, key)
        return float(values.sum())

    def write_shapes(self, region_root_dir: str, output_dir: str, category: str,
                      region_role: str, variable: str,
                      mass_point: Optional[str] = None, year: str = "2024",
                      blind: bool = True,
                      region_dir_override: Optional[str] = None,
                      filename_region: Optional[str] = None) -> str:
        """
        Write the Combine shapes ROOT file for one (category, region) bin,
        copying real per-process histograms from the region ROOT file
        (renamed to Combine's $PROCESS/$PROCESS_$SYSTEMATIC convention).

        Shape systematics are only written for a process/region if
        systematic_applies_to_region() says so (Section 3's gated_by_cut).

        Always writes a "data_obs" histogram: TotalBkg (Asimov) if blind,
        else the real data_obs histogram — the datacard's `observation -1`
        line requires this key to exist in shapes.root regardless of mode.

        region_dir_override, filename_region: see write_datacard's docstring —
        must be passed identically so the datacard and its shapes file agree
        on both which histogram directory they read from and what filename
        they're discoverable under.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        shapes_file = output_path / self._resolve_filename(
            "shapes_file", category, filename_region or region_role, year, mass_point)
        region_dir = region_dir_override or region_dir_from_role(region_role)
        region_key = f"{category}:{region_role}"
        is_sr = region_role == "SR"

        processes = self.datacard_config["processes"]
        process_keys = [p for p in processes.keys()
                        if not processes[p].get("is_signal", False) and processes[p].get("enabled", True)]
        if is_sr:
            process_keys = ["signal"] + process_keys

        with uproot.recreate(str(shapes_file)) as f:
            for process in process_keys:
                proc_config = processes[process]
                datacard_name = proc_config["name"]

                if proc_config.get("is_signal", False) and mass_point is None:
                    raise ValueError("mass_point is required to write signal shapes")
                hist_key = self._process_hist_key(process, mass_point)

                values = load_region_histogram(region_root_dir, category, region_dir,
                                                variable, hist_key)
                edges = load_region_bin_edges(region_root_dir, category, region_dir, variable)
                # Floor exactly-zero bins — a real, reproduced failure mode:
                # binned mode splits each bin into its own single-bin Combine
                # channel, and Combine's ShapeTools.getExtraNorm raises
                # "Null norm for channel X, process Y" on a process with
                # exactly zero yield in that bin (confirmed against real
                # combine 10.6.1: singletop had 0 events in every bin of a Z
                # CR — physically plausible low-yield background, not a data
                # bug). Unbinned mode masks this by summing all bins into one
                # nonzero total, so it never surfaced there. Same ZERO_FLOOR
                # already used for PDF-normalization rescaling (combine_inputs.py).
                values = np.clip(values.astype(float), ZERO_FLOOR, None)
                f[datacard_name] = (values, edges.astype(float))

                for sys_name, sys_config in self.datacard_config["systematics"].items():
                    if sys_config.get("type") != "shape":
                        continue
                    if process not in sys_config.get("processes", []):
                        continue
                    if not systematic_applies_to_region(self.regions_config, region_key,
                                                         sys_config.get("gated_by_cut")):
                        continue

                    syst_suffix = sys_config["syst_suffix"]
                    for direction, out_direction in (("UP", "Up"), ("DOWN", "Down")):
                        try:
                            syst_values = load_region_syst_histogram(
                                region_root_dir, category, region_dir, variable,
                                syst_suffix, direction, hist_key,
                            )
                        except (FileNotFoundError, KeyError):
                            logging.warning(
                                f"Shape variant '{syst_suffix}{direction}' not found for "
                                f"{hist_key} in {category}:{region_dir} — skipping."
                            )
                            continue
                        # Must match _write_systematics_block's row_name exactly
                        # (both derive from the same name_template) — otherwise
                        # the datacard's "$PROCESS_$SYSTEMATIC" row references a
                        # histogram key that doesn't exist in this file.
                        combine_sys_name = self._resolve_systematic_name(sys_name, sys_config, year)
                        out_key = f"{datacard_name}_{combine_sys_name}{out_direction}"
                        syst_values = np.clip(syst_values.astype(float), ZERO_FLOOR, None)
                        f[out_key] = (syst_values, edges.astype(float))

            edges = load_region_bin_edges(region_root_dir, category, region_dir, variable)
            if blind:
                data_values = load_region_histogram(region_root_dir, category, region_dir,
                                                     variable, "TotalBkg")
            else:
                try:
                    data_values = load_region_histogram(region_root_dir, category, region_dir,
                                                         variable, "data_obs")
                except KeyError as exc:
                    raise KeyError(
                        f"Unblind requested for {category}:{region_dir} but 'data_obs' is "
                        f"absent from the region ROOT file — histo-production must be rerun "
                        f"with --show-data for this region."
                    ) from exc
            f["data_obs"] = (data_values.astype(float), edges.astype(float))

        logging.info(f"Shapes file written to {shapes_file}")

        return str(shapes_file)

    def create_workspace(self, datacard_file: str, output_dir: str,
                          workspace_filename: Optional[str] = None) -> str:
        """
        Create Combine workspace from datacard via text2workspace.py.

        workspace_filename: exact output filename, or None to derive one from
        the datacard's own stem (datacard_X.txt -> workspace_X.root) — avoids
        assuming a single fixed workspace.root name now that datacard_file is
        a per-(year,category,region,mass_point) template, not a constant.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        datacard_path = Path(datacard_file)
        if workspace_filename is None:
            datacard_stem = datacard_path.stem
            workspace_filename = datacard_stem.replace("datacard", "workspace", 1) + ".root" \
                if "datacard" in datacard_stem else f"{datacard_stem}_workspace.root"

        workspace_file = output_path / workspace_filename

        cmd = [
            self.advanced_config["combine_commands"]["text2workspace"],
            datacard_path.name,   # basename only — cwd=output_path below, and the
                                   # datacard is always written into output_path
            "-o", workspace_filename,
        ]
        cmd.extend(self.advanced_config.get("workspace_options", {}).get("args", []))

        try:
            # cwd=output_path: the datacard's `shapes *` lines are relative
            # filenames (e.g. 1b_shapes_....root), resolved by text2workspace.py
            # relative to its own working directory — must match where those
            # files actually live (same directory as the datacard itself),
            # matching CombineRunner._run_steps's cwd convention.
            result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                     cwd=str(output_path))
            logging.info(f"Workspace created: {workspace_file}")
            logging.debug(f"text2workspace output: {result.stdout}")
        except FileNotFoundError as e:
            raise RuntimeError(
                "text2workspace.py not found on PATH — Combine is not installed in "
                "this environment. See README for the optional Combine install step."
            ) from e
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create workspace: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        return str(workspace_file)

    @staticmethod
    def _find_one_datacard(directory: Path) -> Path:
        """Find the single .txt datacard in a directory. output.datacard_file
        is a filename TEMPLATE (varies by year/category/region/mass_point),
        so callers that don't know the exact resolved name glob for it
        instead of reconstructing the template."""
        matches = sorted(directory.glob("*.txt"))
        if not matches:
            raise FileNotFoundError(f"No datacard (.txt) found in {directory}")
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Multiple datacard .txt files found in {directory}, expected exactly "
                f"one: {[m.name for m in matches]}"
            )
        return matches[0]

    @staticmethod
    def _read_bin_names(datacard: Path) -> List[str]:
        """Extract a datacard's own bin name(s) from its "bin" line (the
        first one — the "bin" line, not "bin"-prefixed process/rate rows
        further down, which repeat each name once per process). Unbinned
        mode's cards have exactly one bin (SR_1b, Wlnu_1b, ...); binned
        mode's have N (SR_bin1..SR_binN)."""
        with open(datacard) as f:
            for line in f:
                tokens = line.split()
                if tokens and tokens[0] == "bin":
                    return tokens[1:]
        raise ValueError(f"No 'bin' line found in {datacard}")

    def merge_region(self, input_dir: str, output_dir: str, category: str,
                      mass_point: str, control_region_dirs: List[str]) -> str:
        """
        Merge one category's SR (for mass_point) + its CR datacards into one
        region-combined card via combineCards.py — Run2's SR+CR
        bins-in-one-file convention (verified against the real
        bbDM_datacard_run2_... card, whose bin line lists
        d2016_cat_1b_sr_bin1..4, d2016_cat_1b_wl_bin1..4,
        d2016_cat_1b_zll_bin1..4 all in one file). Needed so CR rateParams
        actually constrain the fit (currently they don't — merge_categories
        only ever merged the SR-only 1b/2b cards) and so GoF/pulls
        channel-masking (mask_SR_1b) has CR bins to mask against.

        control_region_dirs is the caller-resolved list of CR output-dir names
        for this category — e.g. ["Wlnu", "Zll"] when combine_emu merges
        e/mu-paired CRs, or the raw ["CR_Wmunu", "CR_Wenu", "CR_Zmumu",
        "CR_Zee"] when combine_emu is off — NOT hardcoded here, since
        make-datacard already resolves this same distinction (see
        merged_region_dir_for_role in combine_inputs.py) and both must agree
        on which CR directories actually exist on disk.

        Unbinned mode (1 bin/card): combineCards.py's {label}=<card> form is
        used, labeled with each card's own existing bin name (SR_1b, Wlnu_1b,
        ...) — reproduced by direct testing that the unlabeled positional
        form discards a single-bin card's original name entirely (renames to
        bare ch1/ch2).

        Binned mode (N bins/card): the SAME labeled form instead PREFIXES
        every bin with "{label}_bin{i}=" per bin (combineCards.py's label
        syntax doesn't support one label covering multiple existing bin
        names cleanly) — reproduced directly: labeling a 4-bin card with its
        first bin's name alone corrupts every other bin's name
        (SR_bin1_SR_bin2, ...). Instead binned mode uses the UNLABELED
        positional form, which (confirmed by direct testing, unlike the
        single-bin case) preserves every original multi-bin name intact,
        merely ch{N}_-prefixing it (ch1_SR_bin1, ch2_Wlnu_bin1, ...) — that
        prefix is then stripped in Python, and rateParam names (not bin
        names) are what ties SR's bin-N to CR's bin-N, so the ch{N}_ prefix
        never affects the tie regardless.
        """
        sr_dir = Path(input_dir) / category / mass_point
        if not sr_dir.is_dir():
            raise FileNotFoundError(f"Missing SR datacard directory for mass point "
                                     f"'{mass_point}': {sr_dir}")
        sr_card = self._find_one_datacard(sr_dir)

        cr_cards: List[Path] = []
        for cr_dir_name in control_region_dirs:
            cr_dir = Path(input_dir) / category / cr_dir_name
            if not cr_dir.is_dir():
                raise FileNotFoundError(f"Missing CR datacard directory '{cr_dir_name}' "
                                         f"for category '{category}': {cr_dir}")
            cr_cards.append(self._find_one_datacard(cr_dir))

        all_cards = [sr_card] + cr_cards
        bin_names_by_card = [self._read_bin_names(c) for c in all_cards]
        is_binned = any(len(names) > 1 for names in bin_names_by_card)

        output_path = Path(output_dir) / category / mass_point
        output_path.mkdir(parents=True, exist_ok=True)
        merged_card = output_path / sr_card.name

        if is_binned:
            # Unlabeled: preserves original multi-bin names (verified), just
            # ch{N}_-prefixed — stripped below. Source-dir labels for shapes
            # filenames come from the card's own directory name (already
            # unique: mass_point for SR, cr_dir_name for each CR) since
            # there's no combineCards.py label to reuse here.
            source_labels = [mass_point] + list(control_region_dirs)
            cmd = ["combineCards.py"] + [str(card) for card in all_cards]
        else:
            source_labels = [self._read_bin_names(c)[0] for c in all_cards]
            cmd = ["combineCards.py"] + [f"{label}={card}" for label, card in zip(source_labels, all_cards)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "combineCards.py not found on PATH — Combine is not installed in "
                "this environment. See README for the optional Combine install step."
            ) from e
        except subprocess.CalledProcessError as e:
            logging.error(f"merge_region failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        merged_text = result.stdout
        for card in all_cards:
            merged_text = merged_text.replace(f"{card.parent}/", "")

        if is_binned:
            # Strip combineCards.py's auto ch{N}_ prefix (N is 1-based,
            # positional order of all_cards) to recover the original names.
            for i in range(len(all_cards)):
                merged_text = merged_text.replace(f"ch{i + 1}_", "")

        # Same fix as merge_categories/merge_eras: rewrite each source card's
        # shapes filename to match the "{label}_" prefix the file is actually
        # copied under below, otherwise text2workspace.py looks for a file
        # that doesn't exist in the merged output directory.
        for label, card in zip(source_labels, all_cards):
            for src_shapes in card.parent.glob("*.root"):
                merged_text = merged_text.replace(src_shapes.name, f"{label}_{src_shapes.name}")

        with open(merged_card, "w") as f:
            f.write(merged_text)

        for label, card in zip(source_labels, all_cards):
            for src_shapes in card.parent.glob("*.root"):
                shutil.copy(src_shapes, output_path / f"{label}_{src_shapes.name}")

        logging.info(f"Region-merged datacard written to {merged_card}")
        return str(merged_card)

    def merge_categories(self, input_dir: str, output_dir: str, mass_point: str) -> str:
        """
        Merge a mass point's per-category datacards (1b + 2b) into one
        combined-category card via combineCards.py.

        Ports Run2's mergeCategoryDatacards.py: combineCards.py cat_1b=<card>
        cat_2b=<card> > out, then strips the cat_1b=/cat_2b= source-path
        prefixes from the merged card's shapes lines.
        """
        dir_1b = Path(input_dir) / "1b" / mass_point
        dir_2b = Path(input_dir) / "2b" / mass_point
        if not dir_1b.is_dir() or not dir_2b.is_dir():
            raise FileNotFoundError(
                f"Missing per-category datacard directory for mass point '{mass_point}': "
                f"{dir_1b} / {dir_2b}"
            )
        card_1b = self._find_one_datacard(dir_1b)
        card_2b = self._find_one_datacard(dir_2b)

        output_path = Path(output_dir) / mass_point
        output_path.mkdir(parents=True, exist_ok=True)
        merged_card = output_path / card_1b.name.replace("_1b_", "_C_") \
            if "_1b_" in card_1b.name else output_path / f"C_{card_1b.name}"

        cmd = ["combineCards.py", f"cat_1b={card_1b}", f"cat_2b={card_2b}"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "combineCards.py not found on PATH — Combine is not installed in "
                "this environment. See README for the optional Combine install step."
            ) from e
        except subprocess.CalledProcessError as e:
            logging.error(f"merge_categories failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        merged_text = result.stdout
        for src_dir in (str(card_1b.parent), str(card_2b.parent)):
            merged_text = merged_text.replace(f"{src_dir}/", "")

        # combineCards.py's output shapes lines still reference each category's
        # ORIGINAL shapes filename (e.g. shapes_2024_1b_SR_....root); the files
        # actually get copied into output_path with a "{label}_" prefix below
        # to avoid a 1b/2b filename collision in one directory. Rewrite the
        # shapes lines to match the names the files are actually copied to —
        # otherwise text2workspace.py looks for a file that doesn't exist there.
        for src_dir, label in ((card_1b.parent, "1b"), (card_2b.parent, "2b")):
            for src_shapes in src_dir.glob("*.root"):
                merged_text = merged_text.replace(src_shapes.name, f"{label}_{src_shapes.name}")

        with open(merged_card, "w") as f:
            f.write(merged_text)

        for src_dir, label in ((card_1b.parent, "1b"), (card_2b.parent, "2b")):
            for src_shapes in src_dir.glob("*.root"):
                shutil.copy(src_shapes, output_path / f"{label}_{src_shapes.name}")

        logging.info(f"Category-merged datacard written to {merged_card}")
        return str(merged_card)

    def merge_eras(self, input_dir: str, output_dir: str, mass_point: str,
                   active_years: List[str]) -> str:
        """
        Merge active eras' category-merged datacards into one full-Run3 card
        via combineCards.py.

        Ports Run2's mergeYearDatacards.py, generalized from a hardcoded
        2016/2017/2018 loop to combine.yaml's configurable eras list.
        """
        era_cards: Dict[str, Path] = {}
        for year in active_years:
            era_dir = Path(input_dir) / str(year) / "C" / mass_point
            if not era_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing category-merged datacard directory for era '{year}', "
                    f"mass point '{mass_point}': {era_dir}"
                )
            era_cards[str(year)] = self._find_one_datacard(era_dir)

        output_path = Path(output_dir) / mass_point
        output_path.mkdir(parents=True, exist_ok=True)
        first_card_name = next(iter(era_cards.values())).name
        merged_card = output_path / (
            first_card_name.replace(next(iter(active_years)), "run3", 1)
            if next(iter(active_years)) in first_card_name else f"run3_{first_card_name}"
        )

        cmd = ["combineCards.py"] + [f"era_{year}={card}" for year, card in era_cards.items()]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "combineCards.py not found on PATH — Combine is not installed in "
                "this environment. See README for the optional Combine install step."
            ) from e
        except subprocess.CalledProcessError as e:
            logging.error(f"merge_eras failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        merged_text = result.stdout
        for card in era_cards.values():
            merged_text = merged_text.replace(f"{card.parent}/", "")

        # Same fix as merge_categories(): rewrite each era's shapes filename to
        # match the "{year}_" prefix the file is actually copied under below,
        # otherwise text2workspace.py looks for a file that doesn't exist.
        for year, card in era_cards.items():
            for src_shapes in card.parent.glob("*.root"):
                merged_text = merged_text.replace(src_shapes.name, f"{year}_{src_shapes.name}")

        with open(merged_card, "w") as f:
            f.write(merged_text)

        for year, card in era_cards.items():
            for src_shapes in card.parent.glob("*.root"):
                shutil.copy(src_shapes, output_path / f"{year}_{src_shapes.name}")

        logging.info(f"Era-merged datacard written to {merged_card}")
        return str(merged_card)


class CombineRunner:
    """
    Runs Combine fits and analyses, driven entirely by command templates in
    combine.yaml's advanced.commands block — no -M <mode>, flag, or step
    sequence is hardcoded here.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fit_config = config["fit"]
        self.output_config = config["output"]
        self.advanced_config = config["advanced"]

    def _run_steps(self, mode: str, workspace: str, output_dir: str,
                    blind: bool, extra_format_args: Optional[Dict[str, Any]] = None) -> None:
        """Format and execute every step of a command template in order.
        Every step runs with cwd=output_dir (Combine's own output files —
        higgsCombine*.root — are always written relative to cwd), so
        `workspace` is passed to the command as a basename, not the full
        path, to avoid double-nesting it under output_dir. This requires the
        workspace file to actually live in output_dir (true for every current
        caller: create_workspace() always writes there)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        workspace_arg = Path(workspace).name if Path(workspace).parent == output_path \
            else str(workspace)

        commands = self.advanced_config["commands"]
        if mode not in commands:
            raise KeyError(f"No command template for mode '{mode}' in combine.yaml advanced.commands")

        general_args = commands.get("general_args", [])
        blind_args = ["-t", "-1"] if blind else []

        format_args = {
            "workspace": workspace_arg,
            "output_dir": str(output_path),
            "blind_args": " ".join(blind_args),
            "n_workers": self.advanced_config.get("parallel", {}).get("n_workers", 4),
            "gof_algo": self.fit_config.get("options", {}).get("goodness_of_fit", {}).get("algorithm", "saturated"),
            "toys": self.fit_config.get("options", {}).get("goodness_of_fit", {}).get("toys", 500),
            "run_mode": "expected",
        }
        if extra_format_args:
            format_args.update(extra_format_args)

        for step in commands[mode]["steps"]:
            binary_key = step["binary"]
            if binary_key in self.advanced_config["combine_commands"]:
                binary = self.advanced_config["combine_commands"][binary_key]
            elif binary_key == "combine":
                binary = self.advanced_config["combine_commands"]["combine"]
            elif binary_key == "combineTool.py":
                binary = self.advanced_config["combine_commands"]["combine_tool"]
            else:
                binary = binary_key

            args = []
            for token in step["args"]:
                formatted = token.format(**format_args)
                if formatted == "":
                    continue
                args.extend(formatted.split(" "))

            cmd = [binary] + [a for a in args if a]
            cmd.extend(general_args)

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                         cwd=str(output_path))
                logging.info(f"{mode} step completed: {' '.join(cmd)}")
                logging.debug(f"Combine output: {result.stdout}")
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"'{binary}' not found on PATH — Combine is not installed in this "
                    f"environment. See README for the optional Combine install step."
                ) from e
            except subprocess.CalledProcessError as e:
                logging.error(f"{mode} step failed: {' '.join(cmd)}")
                logging.error(f"Error output: {e.stderr}")
                raise

    def run_asymptotic_limits(self, datacard_or_workspace: str, output_dir: str,
                               blind: bool = True) -> str:
        """Returns the actual combine output file — same fix as
        run_goodness_of_fit: combine's real output filename is
        higgsCombineTest.AsymptoticLimits.mH120*.root (its default -n Test
        label, no --seed passed here), not the output_config's
        "asymptotic_limits.root" name, which is never actually created."""
        self._run_steps("AsymptoticLimits", datacard_or_workspace, output_dir, blind)
        if not blind:
            self._run_steps("AsymptoticLimits", datacard_or_workspace, output_dir, blind,
                             extra_format_args={"run_mode": "observed"})
        matches = sorted(Path(output_dir).glob("higgsCombineTest.AsymptoticLimits.mH120*.root"))
        if not matches:
            raise FileNotFoundError(
                f"No higgsCombineTest.AsymptoticLimits.mH120*.root found in {output_dir} "
                f"after running AsymptoticLimits."
            )
        return str(matches[0])

    def run_fit_diagnostics(self, datacard_or_workspace: str, output_dir: str,
                             blind: bool = True) -> str:
        """Same fix as run_goodness_of_fit/run_asymptotic_limits: combine's
        default -n Test label produces higgsCombineTest.FitDiagnostics.mH120.root
        (or fitDiagnosticsTest.root, depending on combine version), not the
        output_config's "fitDiagnostics.root" name."""
        self._run_steps("FitDiagnostics", datacard_or_workspace, output_dir, blind)
        matches = sorted(Path(output_dir).glob("fitDiagnostics*.root"))
        if not matches:
            raise FileNotFoundError(
                f"No fitDiagnostics*.root found in {output_dir} after running FitDiagnostics."
            )
        return str(matches[0])

    @staticmethod
    def _sr_bin_names(datacard_file: str) -> List[str]:
        """Extract SR bin names from a datacard's "bin" line. SR bin names
        always contain "SR" as a distinct name-component (region_dir is
        literally "SR"): "SR_1b" (single region-merged card), "cat_1b_SR_1b"
        (after category-merge), etc. — matched via "_SR_"/leading "SR_"/
        trailing "_SR" rather than a loose substring check, so a CR whose
        name happened to contain "SR" elsewhere wouldn't false-positive.
        Used to build GoodnessOfFit's channel-masking parameters
        (mask_<bin>=1/0) without hardcoding bin names, which vary by
        category/era/merge stage."""
        with open(datacard_file) as f:
            for line in f:
                tokens = line.split()
                if tokens and tokens[0] == "bin":
                    bin_names = tokens[1:]
                    return [b for b in bin_names
                            if b == "SR" or b.startswith("SR_") or b.endswith("_SR")
                            or "_SR_" in b]
        raise ValueError(f"No 'bin' line found in {datacard_file}")

    def run_goodness_of_fit(self, datacard_or_workspace: str, output_dir: str,
                             datacard_file: str, blind: bool = True) -> str:
        """Run GoF as two combine calls (-n Observed, -n Toys — see
        combine.yaml's GoodnessOfFit template), matching Run2's actual
        makeGOF_allAlgos.sh pattern: nuisances fit with SR masked (CR-only),
        test statistic evaluated with SR unmasked, signal strength frozen at
        0 — NOT a plain unmasked fit, which would let SR data itself pull the
        background normalization instead of testing whether the CR-derived
        background model actually describes SR. `blind` is accepted for
        interface consistency with the other run_* methods but doesn't gate
        GoF's own combine invocation — GoF always compares an observed test
        statistic against a background-only toy distribution regardless of
        SR blinding; `blind` only gates the datacard's observation/data_obs,
        upstream of this call.

        datacard_file: the .txt datacard used to build datacard_or_workspace
        (the .root workspace itself doesn't expose bin names as plain text) —
        used to derive the SR bin names for channel-masking.

        Returns the observed-step output file; run-all/CLI callers pass this
        plus the toy-step file (same output_dir, "Toys" label) to
        parse_results()."""
        sr_bins = self._sr_bin_names(datacard_file)
        if not sr_bins:
            raise ValueError(f"No SR bin found in {datacard_file} — cannot build "
                              f"GoodnessOfFit's channel-masking parameters.")
        mask_sr_on = ",".join(f"mask_{b}=1" for b in sr_bins)
        mask_sr_off = ",".join(f"mask_{b}=0" for b in sr_bins)

        self._run_steps("GoodnessOfFit", datacard_or_workspace, output_dir, blind,
                         extra_format_args={"mask_sr_on": mask_sr_on, "mask_sr_off": mask_sr_off})
        # combine appends a seed suffix to its output filename whenever --seed
        # is passed (mH120.<seed>.root, not a fixed mH120.root) — glob instead
        # of assuming an exact name, since the seed is yaml-configurable.
        matches = sorted(Path(output_dir).glob("higgsCombineObserved.GoodnessOfFit.mH120*.root"))
        if not matches:
            raise FileNotFoundError(
                f"No higgsCombineObserved.GoodnessOfFit.mH120*.root found in {output_dir} "
                f"after running GoodnessOfFit's Observed step."
            )
        return str(matches[0])

    def run_collect_goodness_of_fit(self, observed_file: str, output_dir: str) -> str:
        """Merge the Observed + Toys GoodnessOfFit outputs into one .json via
        combineTool.py -M CollectGoodnessOfFit, matching Run2's real
        makeGOF_allAlgos.sh (`combineTool.py -M CollectGoodnessOfFit --input
        higgsCombine*.GoodnessOfFit.mH120.root higgsCombine*_toy...root -o
        gof....json`) — replaces the hand-rolled p-value computation that used
        to live in _parse_goodness_of_fit with Combine's own tool, same as
        Run2 does, rather than re-deriving toy-comparison logic in Python.

        observed_file: run_goodness_of_fit's return value (the Observed
        step's output); the Toys step's output is found alongside it via glob.
        """
        toys_matches = sorted(Path(observed_file).parent.glob(
            "higgsCombineToys.GoodnessOfFit.mH120*.root"))
        if not toys_matches:
            raise FileNotFoundError(
                f"No higgsCombineToys.GoodnessOfFit.mH120*.root found alongside "
                f"{observed_file} — run the Toys step before collecting."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_json = output_path / "gof.json"

        cmd = [
            self.advanced_config["combine_commands"]["combine_tool"],
            "-M", "CollectGoodnessOfFit",
            "--input", str(Path(observed_file).name), str(toys_matches[0].name),
            "-m", "120.0",
            "-o", output_json.name,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(output_path))
        except FileNotFoundError as e:
            raise RuntimeError(
                "combineTool.py not found on PATH — Combine is not installed in "
                "this environment. See README for the optional Combine install step."
            ) from e
        except subprocess.CalledProcessError as e:
            logging.error(f"CollectGoodnessOfFit failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        logging.info(f"GoodnessOfFit collected to {output_json}")
        return str(output_json)

    def run_plot_gof(self, gof_json: str, output_dir: str, algo: str = "saturated",
                      title_right: str = "", mass_point: str = "") -> str:
        """Render gof.json matching plotGOF_fromDanyer.py exactly (ported
        line-for-line from bbdmRun2/bbDMlimitmodelrateParam_oneRP/
        plotGOF_fromDanyer.py's ROOT/PyROOT logic to matplotlib): light-green
        filled histogram (alpha 0.20) with a darker outline, a red-shaded
        overlay on bins at/above the observed value's bin (not a separate
        line), a red arrow marking the observed value, "CMS Internal" logo,
        p-value/toys text boxes, legend. Verified against the source script
        directly, not guessed from the reference image alone — that image's
        apparent per-bin tick marks are NOT drawn by this script (no error
        bars anywhere in plotGOF_fromDanyer.py); only bin-content silhouette,
        red arrow, and legend are real.

        Histogram range: [0.85*toys.min(), 0.95*toys.max()] — asymmetric,
        matching the source's intent. The source additionally clamped the
        upper bound to a literal 100 (a Run2-specific magic number tuned to
        their toy-statistic scale, ~30 max there); dropped here since our
        toy distributions naturally range up to ~150 and a literal 100 would
        truncate the real tail rather than protecting against an outlier.

        mass_point is embedded in the output FILENAME (gof_{mass_point}.pdf/png)
        per user request; title_right is drawn INSIDE the plot (e.g.
        "S+B hypothesis(1b+2b 2024)") and does not include the mass point.
        """
        import matplotlib.pyplot as plt
        try:
            import mplhep as hep
            hep.style.use(hep.style.CMS)
            has_mplhep = True
        except ImportError:
            has_mplhep = False

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = f"_{mass_point}" if mass_point else ""
        out_stem = output_path / f"gof{suffix}"

        with open(gof_json) as f:
            data = json.load(f)
        mass_key = next(iter(data.keys()))
        entry = data[mass_key]
        obs = entry["obs"][0] if isinstance(entry["obs"], list) else entry["obs"]
        toys = np.asarray(entry["toy"], dtype=float)
        n_toys = len(toys)

        n_bins = 100
        x_min = 0.85 * toys.min()
        x_max = 0.95 * toys.max()
        raw_counts, edges = np.histogram(toys, bins=n_bins, range=(x_min, x_max))
        norm = raw_counts.sum() if raw_counts.sum() > 0 else 1
        counts = raw_counts / norm

        obs_bin = int(np.searchsorted(edges, obs, side="right")) - 1
        p_value = counts[obs_bin:].sum() if 0 <= obs_bin < n_bins else (
            1.0 if obs_bin < 0 else 0.0)

        fig, ax = plt.subplots(figsize=(8.5, 8.7))

        ax.stairs(counts, edges, fill=True, color="#c8eab4", edgecolor="#7ce03c",
                  linewidth=1.3, label="Expected (Toys)")

        # Thin per-bin vertical tick above each bin top — a real feature of
        # the reference plot (confirmed against a real screenshot of
        # plotGOF_fromDanyer.py's actual rendered output, not just its
        # source code), drawn as a short line segment per bin rather than a
        # Poisson error bar (the source script draws no error bars at all).
        tick_height = 0.05 * counts.max()
        for i in range(n_bins):
            if counts[i] <= 0:
                continue
            xc = 0.5 * (edges[i] + edges[i + 1])
            ax.plot([xc, xc], [counts[i], counts[i] + tick_height],
                    color="#7ce03c", linewidth=0.8)

        tail_counts = np.where(np.arange(n_bins) >= obs_bin, counts, 0.0)
        if obs_bin < n_bins:
            ax.stairs(tail_counts, edges, fill=True, color="#cc0000", alpha=0.40,
                      linewidth=0)

        ax.set_ylim(0, 1.6 * counts.max())
        ax.set_xlim(x_min, x_max)

        obs_in_range = 0 <= obs_bin < n_bins
        if obs_in_range:
            arrow_x = edges[obs_bin]
            y_top = 0.4 * counts[obs_bin]
            ax.annotate("", xy=(arrow_x, 0.002), xytext=(arrow_x, max(y_top, 0.01)),
                        arrowprops=dict(arrowstyle="-|>", color="#cc0000", lw=3,
                                         mutation_scale=20),
                        annotation_clip=False)
            arrow_handle = plt.Line2D([0], [0], color="#cc0000", lw=3, label="Observed")
        else:
            ax.text(0.5, 0.5, "Observed value not in range", color="#cc0000",
                     fontsize=18, fontweight="bold", ha="center", va="center",
                     transform=ax.transAxes)
            arrow_handle = plt.Line2D([0], [0], color="#cc0000", lw=3, label="Observed")

        ax.set_xlabel(r"$-2\ \ln\lambda$ (%s)" % algo, fontsize=22, loc="right")
        ax.set_ylabel("Normalized to unity", fontsize=22, loc="top")
        ax.tick_params(direction="in", top=True, right=True, which="major",
                        length=7, width=1.0, labelsize=18)
        ax.tick_params(direction="in", top=True, right=True, which="minor",
                        length=4, width=0.8)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

        if has_mplhep:
            hep.cms.label(llabel="Internal", data=False, com=13.6, loc=0, ax=ax)
        else:
            ax.text(0.0, 1.02, "CMS", transform=ax.transAxes, fontsize=20,
                    fontweight="bold", ha="left", va="bottom")
            ax.text(0.115, 1.02, "Internal", transform=ax.transAxes, fontsize=18,
                    fontstyle="italic", ha="left", va="bottom")
            ax.text(1.0, 1.02, "13.6 TeV", transform=ax.transAxes, fontsize=14,
                     ha="right", va="bottom")
        right_x = 0.92
        line_step = 0.08
        y0 = 0.92
        if title_right:
            ax.text(right_x, y0, title_right, transform=ax.transAxes, fontsize=15,
                     ha="right", va="top")

        ax.text(right_x, y0 - line_step, f"{algo.capitalize()}, {n_toys} Toys", transform=ax.transAxes,
                 fontsize=18, fontweight="bold", ha="right", va="top")
        ax.text(right_x, y0 - 2 * line_step, f"p-value: {p_value:.3f}", transform=ax.transAxes,
                 fontsize=19, fontstyle="italic", ha="right", va="top")

        fill_handle = plt.Rectangle((0, 0), 1, 1, fc="#c8eab4", ec="#7ce03c",
                                     linewidth=1.3, label="Expected (Toys)")
        ax.legend(handles=[arrow_handle, fill_handle], loc="upper right",
                  bbox_to_anchor=(right_x, y0 - 2.5 * line_step), frameon=False, fontsize=15)

        fig.tight_layout()
        fig.savefig(f"{out_stem}.pdf")
        fig.savefig(f"{out_stem}.png", dpi=150)
        plt.close(fig)

        return f"{out_stem}.pdf"

    _PULLS_MODES = {
        "CRonly": ("PullsCRonly", "Data"),
        "asimov_t0": ("PullsAsimovT0", "Asimov"),
        "sb_t0": ("PullsSbT0", "Data"),
        "sb_t1": ("PullsSbT1", "Data"),
    }

    def _resolve_lumi_text(self, year: str) -> str:
        """Build PlotPulls.C's lumi label from combine.yaml's eras[].year_config
        lumi value — Run2's PlotPulls.C hardcoded a 2016/2017/2018/run2
        filename-substring lookup table; Run3's eras aren't fixed at
        macro-write time the way Run2's 3-era list was, so this is resolved
        from config instead."""
        for era in self.config["eras"]:
            if str(era["year"]) == str(year):
                try:
                    with open(era["year_config"]) as f:
                        year_cfg = yaml.safe_load(f)
                    lumi = year_cfg.get("lumi")
                    if lumi is not None:
                        return f"{float(lumi):.2f} fb^{{-1}} ({year})"
                except FileNotFoundError:
                    pass
                break
        return ""

    def run_pulls(self, workspace: str, output_dir: str, datacard_file: str,
                   mode: str, year: str = "2024") -> str:
        """Run one of Run2's 4 pulls modes (CRonly/asimov_t0/sb_t0/sb_t1,
        pulls_oneRP.sh) via FitDiagnostics.

        CRonly is a DIFFERENT tooling path from the other 3 modes — verified
        by reading pulls_oneRP.sh in full: with SR masked, FitDiagnostics
        only produces a fit_b RooFitResult (no S+B fit is meaningful without
        a signal region), so CRonly's post-fit values are read directly via
        plotPostNuisance_combine.C (fit_b -> post-fit central+uncertainty
        plot), NOT diffNuisances.py (which requires fit_s and raises
        "does not contain the output of the signal fit 'fit_s'" otherwise —
        reproduced against real combine 10.6.1). asimov_t0/sb_t0/sb_t1 run
        diffNuisances.py -g (pull extraction into a ROOT "nuisances"
        TCanvas) then PlotPulls.C (paginated CMS-style plot).

        mode must be one of _PULLS_MODES' keys. CRonly needs SR masked
        (derived from datacard_file's own bin names, same as GoodnessOfFit);
        the other 3 modes run unmasked with different -t/--expectSignal
        combinations.

        Returns the first page's .pdf path (paginated — see PlotPulls.C/
        plotPostNuisance_combine.C for the full file set when there are
        >89-90 nuisances).
        """
        if mode not in self._PULLS_MODES:
            raise ValueError(f"Unknown pulls mode '{mode}', expected one of {list(self._PULLS_MODES)}")
        command_name, data_label = self._PULLS_MODES[mode]

        extra_format_args: Dict[str, Any] = {}
        if mode == "CRonly":
            sr_bins = self._sr_bin_names(datacard_file)
            if not sr_bins:
                raise ValueError(f"No SR bin found in {datacard_file} — cannot mask SR "
                                  f"for the CRonly pulls mode.")
            extra_format_args["mask_sr_on"] = ",".join(f"mask_{b}=1" for b in sr_bins)

        self._run_steps(command_name, workspace, output_dir, blind=False,
                         extra_format_args=extra_format_args)

        fd_matches = sorted(Path(output_dir).glob("fitDiagnostics_*.root"))
        if not fd_matches:
            raise FileNotFoundError(
                f"No fitDiagnostics_*.root found in {output_dir} after running {command_name}."
            )
        fit_diagnostics_file = fd_matches[0]

        # Persist a stable per-mode copy — Run2 keeps every mode's
        # fitDiagnostics_{catg}_{year}_{mode}_{dirname}.root under
        # fitDiagnosticsDir/ (pulls_oneRP.sh's --out flag) rather than letting
        # each mode's run overwrite the last one, since all 4 are meant to be
        # inspectable afterward, not just the final mode run.
        fit_diagnostics_dir = Path(output_dir) / "fitDiagnosticsDir"
        fit_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        persisted_fd = fit_diagnostics_dir / f"fitDiagnostics_{mode}.root"
        shutil.copy(fit_diagnostics_file, persisted_fd)

        pulls_tooling = self.advanced_config.get("pulls_tooling", {})
        lumi_text = self._resolve_lumi_text(year)

        if mode == "CRonly":
            macro = pulls_tooling.get("plot_postnuisance_macro",
                                       "condorJobs/combine/plotPostNuisance_combine.C")
            root_cmd = [
                "root", "-l", "-b", "-q",
                f'{macro}("{fit_diagnostics_file}", "{output_dir}/", "{mode}", "{lumi_text}")',
            ]
            try:
                subprocess.run(root_cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError as e:
                raise RuntimeError(
                    "root not found on PATH — Combine/ROOT is not installed in this environment."
                ) from e
            except subprocess.CalledProcessError as e:
                logging.error(f"plotPostNuisance_combine.C failed: {e}")
                logging.error(f"Error output: {e.stderr}")
                raise
        else:
            diff_nuisances_bin = self.advanced_config["combine_commands"].get(
                "diff_nuisances", "diffNuisances.py")
            diff_nuisances_args = pulls_tooling.get("diff_nuisances_args", ["--abs", "--all"])

            pulls_root = Path(output_dir) / f"pulls_{mode}.root"
            cmd = [diff_nuisances_bin, str(fit_diagnostics_file)] + list(diff_nuisances_args) \
                + ["-g", str(pulls_root)]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError as e:
                raise RuntimeError(
                    "diffNuisances.py not found on PATH — fetched automatically by "
                    "INSTALL_COMBINE=1 source local_setup.sh; see README."
                ) from e
            except subprocess.CalledProcessError as e:
                logging.error(f"diffNuisances.py failed: {e}")
                logging.error(f"Error output: {e.stderr}")
                raise

            macro = pulls_tooling.get("plot_pulls_macro", "condorJobs/combine/PlotPulls.C")
            root_cmd = [
                "root", "-l", "-b", "-q",
                f'{macro}("{pulls_root}", "{output_dir}/", "", "{lumi_text}", "{data_label}")',
            ]
            try:
                subprocess.run(root_cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError as e:
                raise RuntimeError(
                    "root not found on PATH — Combine/ROOT is not installed in this environment."
                ) from e
            except subprocess.CalledProcessError as e:
                logging.error(f"PlotPulls.C failed: {e}")
                logging.error(f"Error output: {e.stderr}")
                raise

        plot_matches = sorted(Path(output_dir).glob(f"pulls_{mode}_*.pdf"))
        if not plot_matches:
            raise FileNotFoundError(
                f"No pulls_{mode}_*.pdf found in {output_dir} after running the pulls plotting step."
            )
        return str(plot_matches[0])

    def run_impacts(self, datacard_or_workspace: str, output_dir: str,
                     blind: bool = True) -> str:
        results_file = Path(output_dir) / self.output_config["fit_results"].get(
            "impacts", "impacts.json")
        self._run_steps("Impacts", datacard_or_workspace, output_dir, blind)
        return str(results_file)

    def run_plot_impacts(self, impacts_json: str, output_dir: str) -> str:
        """Render impacts.json via the official plotImpacts.py (confirmed
        installed and working, no CombineHarvester dependency in this build),
        matching Run2's real impacts.sh (`plotImpacts.py -i impacts_....json
        -o impacts_..._${dirname}`) rather than a custom matplotlib plot."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        out_stem = output_path / "impacts"

        cmd = [
            self.advanced_config["combine_commands"].get("plot_impacts", "plotImpacts.py"),
            "-i", str(impacts_json),
            "-o", str(out_stem),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "plotImpacts.py not found on PATH — Combine is not installed in "
                "this environment. See README for the optional Combine install step."
            ) from e
        except subprocess.CalledProcessError as e:
            logging.error(f"plotImpacts.py failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        return f"{out_stem}.pdf"

    @staticmethod
    def _parse_mass_point_label(mass_point: str) -> Optional[Dict[str, float]]:
        """Parse "MH3_<x>_MH4_<y>_Mchi_<z>" into numeric fields. Returns None
        if the label doesn't match (so callers can skip non-conforming
        entries rather than crash)."""
        m = re.match(r"^MH3_([\d.]+)_MH4_([\d.]+)_Mchi_([\d.]+)$", mass_point)
        if not m:
            return None
        return {"MH3": float(m.group(1)), "MH4": float(m.group(2)), "Mchi": float(m.group(3))}

    def collect_limits(self, card_dir: str, mass_points: List[str],
                        output_dir: str, output_name: str) -> str:
        """Aggregate every mass point's AsymptoticLimits result into one
        combined summary table, matching Run2's actual limits_bbDM_C_2018.txt
        format (verified against bbDMlimitmodelrateParam_oneRP/limits/*/*.txt):
        one row per mass point, columns "MH3 MH4 exp-2s exp-1s exp exp+1s
        exp+2s observed", sorted by (MH3, MH4). Also writes a companion .root
        with TGraphAsymmErrors (exp2/exp1/expmed/obs), matching Run2's
        limits_bbDM_C_2018.root, for Brazil-band plotting.

        card_dir: directory containing {mass_point}/higgsCombineTest.
            AsymptoticLimits.mH120*.root for each mass point (i.e. the same
            dir make-datacard/merge-categories wrote per-mass-point subdirs
            into).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        rows = []
        for mass_point in mass_points:
            parsed = self._parse_mass_point_label(mass_point)
            if parsed is None:
                logging.warning(f"collect_limits: skipping non-conforming mass point label '{mass_point}'")
                continue

            matches = sorted(Path(card_dir).glob(
                f"{mass_point}/higgsCombineTest.AsymptoticLimits.mH120*.root"))
            if not matches:
                logging.warning(f"collect_limits: no AsymptoticLimits output for '{mass_point}' — skipping")
                continue

            limits = self._parse_asymptotic_limits(str(matches[0]))
            required = ("expected_minus_2sigma", "expected_minus_1sigma", "expected",
                        "expected_plus_1sigma", "expected_plus_2sigma")
            if not all(k in limits for k in required):
                logging.warning(f"collect_limits: incomplete limit quantiles for '{mass_point}' — skipping")
                continue

            rows.append({
                "MH3": parsed["MH3"], "MH4": parsed["MH4"],
                "exp_m2": limits["expected_minus_2sigma"],
                "exp_m1": limits["expected_minus_1sigma"],
                "exp": limits["expected"],
                "exp_p1": limits["expected_plus_1sigma"],
                "exp_p2": limits["expected_plus_2sigma"],
                "observed": limits.get("observed"),
            })

        rows.sort(key=lambda r: (r["MH3"], r["MH4"]))

        txt_file = output_path / f"{output_name}.txt"
        with open(txt_file, "w") as f:
            for r in rows:
                obs = r["observed"] if r["observed"] is not None else 0.0
                f.write(f"{r['MH3']:g} {r['MH4']:g} {r['exp_m2']:.4f} {r['exp_m1']:.4f} "
                        f"{r['exp']:.4f} {r['exp_p1']:.4f} {r['exp_p2']:.4f} {obs:.4f}\n")

        root_file = output_path / f"{output_name}.root"
        x = np.array([r["MH4"] for r in rows], dtype=float)
        exp_m2 = np.array([r["exp_m2"] for r in rows], dtype=float)
        exp_m1 = np.array([r["exp_m1"] for r in rows], dtype=float)
        exp = np.array([r["exp"] for r in rows], dtype=float)
        exp_p1 = np.array([r["exp_p1"] for r in rows], dtype=float)
        exp_p2 = np.array([r["exp_p2"] for r in rows], dtype=float)
        obs = np.array([r["observed"] if r["observed"] is not None else 0.0 for r in rows], dtype=float)

        # uproot 5.6's as_TGraph only serializes plain TGraph (TGraphErrors/
        # TGraphAsymmErrors raise NotImplementedError("FIXME") on write), so
        # the Brazil-band graphs (which need asymmetric errors) are written
        # via PyROOT instead, matching Run2's actual TGraphAsymmErrors output.
        import ROOT

        zeros = np.zeros_like(x)
        root_out = ROOT.TFile(str(root_file), "RECREATE")
        n = len(x)
        g_exp2 = ROOT.TGraphAsymmErrors(n, x, exp, zeros, zeros, exp - exp_m2, exp_p2 - exp)
        g_exp2.SetName("exp2")
        g_exp2.Write()
        g_exp1 = ROOT.TGraphAsymmErrors(n, x, exp, zeros, zeros, exp - exp_m1, exp_p1 - exp)
        g_exp1.SetName("exp1")
        g_exp1.Write()
        g_expmed = ROOT.TGraph(n, x, exp)
        g_expmed.SetName("expmed")
        g_expmed.Write()
        g_obs = ROOT.TGraph(n, x, obs)
        g_obs.SetName("obs")
        g_obs.Write()
        root_out.Close()

        logging.info(f"Limits summary written to {txt_file} and {root_file}")
        return str(txt_file)

    def plot_limits(self, limits_txt: str, xsection_json: str, model_key: str,
                     output_dir: str, output_name: str, lumi: Optional[float] = None,
                     model_labels: Optional[Dict[str, str]] = None) -> List[str]:
        """Render one Brazil-band exclusion plot per distinct MH3 value found
        in a collect_limits() .txt summary — matching the reference CMS-style
        plot (Limitplotter_withbands.ipynb / Figure_007.pdf): observed solid,
        median expected dashed, 68%/95% expected bands, theory cross-section
        dash-dot, x-axis = MH4 ("m_a"), y-axis = 95% CL cross section (fb).

        collect_limits() writes r (signal-strength) limits, not absolute
        cross sections — unlike Run2's notebook (which hardcoded a
        per-point theory xsec array), this converts r -> sigma_95%CL using
        the real per-mass-point theory cross section from xsection_signal.json
        (pb -> fb, x1000), so nothing here is hardcoded per mass point.

        Returns the list of PDF paths written (one per MH3 slice).
        """
        import matplotlib.pyplot as plt
        try:
            import mplhep as hep
            hep.style.use("CMS")
            has_mplhep = True
        except ImportError:
            has_mplhep = False

        from .combine_inputs import load_signal_grid
        xsec_grid_pb = load_signal_grid(xsection_json, model_key)

        rows = []
        with open(limits_txt) as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                mh3, mh4, exp_m2, exp_m1, exp, exp_p1, exp_p2, obs = map(float, tokens)
                label = f"MH3_{mh3:g}_MH4_{mh4:g}_Mchi_1"
                if label not in xsec_grid_pb:
                    logging.warning(f"plot_limits: no theory cross section for '{label}' — skipping")
                    continue
                xsec_fb = xsec_grid_pb[label] * 1000.0
                rows.append({
                    "MH3": mh3, "MH4": mh4, "xsec_fb": xsec_fb,
                    "exp_m2": exp_m2 * xsec_fb, "exp_m1": exp_m1 * xsec_fb,
                    "exp": exp * xsec_fb, "exp_p1": exp_p1 * xsec_fb, "exp_p2": exp_p2 * xsec_fb,
                    "obs": obs * xsec_fb if obs > 0 else None,
                })

        mh3_values = sorted({r["MH3"] for r in rows})
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_files = []

        for mh3 in mh3_values:
            slice_rows = sorted((r for r in rows if r["MH3"] == mh3), key=lambda r: r["MH4"])
            if len(slice_rows) < 2:
                logging.warning(f"plot_limits: MH3={mh3:g} has only {len(slice_rows)} point(s) — "
                                 f"skipping (need >=2 for a band plot).")
                continue

            ma = np.array([r["MH4"] for r in slice_rows])
            exp = np.array([r["exp"] for r in slice_rows])
            exp_m1 = np.array([r["exp_m1"] for r in slice_rows])
            exp_p1 = np.array([r["exp_p1"] for r in slice_rows])
            exp_m2 = np.array([r["exp_m2"] for r in slice_rows])
            exp_p2 = np.array([r["exp_p2"] for r in slice_rows])
            theory = np.array([r["xsec_fb"] for r in slice_rows])
            has_obs = all(r["obs"] is not None for r in slice_rows)
            obs = np.array([r["obs"] for r in slice_rows]) if has_obs else None

            fig, ax = plt.subplots(figsize=(8, 8))
            if has_mplhep:
                hep.cms.label(text="", lumi=lumi, loc=0, llabel="", ax=ax, com=13.6)

            if has_obs:
                ax.plot(ma, obs, linestyle="solid", color="black", linewidth=2, label="Observed")
            ax.plot(ma, exp, linestyle="dashed", color="black", label="Median expected")
            ax.fill_between(ma, exp_m2, exp_p2, alpha=0.9, color="#F5BB54", label=r"95% expected")
            ax.fill_between(ma, exp_m1, exp_p1, alpha=0.9, color="#607641", label=r"68% expected")
            ax.plot(ma, theory, color="red", linestyle="-.", alpha=0.7, label=r"$\sigma_{theory}$")

            ax.set_yscale("log")
            ax.set_xlabel(r"$m_{a}$ (GeV)")
            ax.set_ylabel(r"95% CL $\sigma(pp\rightarrow b\bar{b}\chi\bar{\chi})$ (fb)")
            ax.legend(frameon=False, fontsize=14, loc="upper right")

            # Each line gets its own $...$ block — mathtext (matplotlib's
            # built-in math renderer, no external LaTeX install required)
            # does not span math mode across a single string's embedded
            # newlines, which garbled/overlapped the text when combined here.
            heading = (model_labels or {}).get("heading", "2HDM+a")
            body_lines = (model_labels or {}).get("body_lines", [
                r"$b\bar{b}+p_T^{miss}$",
                rf"$m_A$ = {mh3:g} GeV",
                r"$m_{\chi}$ = 1 GeV",
                r"tan$\beta$ = 35",
                r"sin$\theta$ = 0.7",
            ])
            ax.text(0.03, 0.93, heading, transform=ax.transAxes, fontsize=16,
                    weight="bold", va="top")
            y0 = 0.86
            for i, line in enumerate(body_lines):
                ax.text(0.03, y0 - i * 0.045, line, transform=ax.transAxes,
                        fontsize=12, va="top")

            plot_file = output_path / f"{output_name}_MH3_{mh3:g}.pdf"
            fig.savefig(plot_file, bbox_inches="tight")
            plt.close(fig)
            plot_files.append(str(plot_file))
            logging.info(f"Limit plot written to {plot_file}")

        return plot_files

    def parse_results(self, results_file: str, mode: str) -> Dict[str, Any]:
        if mode == "AsymptoticLimits":
            return self._parse_asymptotic_limits(results_file)
        elif mode == "FitDiagnostics":
            return self._parse_fit_diagnostics(results_file)
        elif mode == "GoodnessOfFit":
            return self._parse_goodness_of_fit(results_file)
        elif mode == "Impacts":
            return self._parse_impacts(results_file)
        raise ValueError(f"Unknown Combine mode: {mode}")

    def _parse_asymptotic_limits(self, results_file: str) -> Dict[str, Any]:
        with uproot.open(results_file) as f:
            tree = f["limit"]
            limits = tree["limit"].array(library="np")
            quantiles = tree["quantileExpected"].array(library="np")

        result = {}
        quantile_map = {
            -1.0: "observed",
            0.025: "expected_minus_2sigma",
            0.16: "expected_minus_1sigma",
            0.5: "expected",
            0.84: "expected_plus_1sigma",
            0.975: "expected_plus_2sigma",
        }
        for q, name in quantile_map.items():
            idx = np.argmin(np.abs(quantiles - q))
            if abs(quantiles[idx] - q) < 1e-3:
                result[name] = float(limits[idx])
        return result

    def _parse_fit_diagnostics(self, results_file: str) -> Dict[str, Any]:
        with uproot.open(results_file) as f:
            tree_name = "tree_fit_sb" if "tree_fit_sb" in f else "tree_fit_b"
            tree = f[tree_name]
            r = float(tree["r"].array(library="np")[0])
            r_err = float(tree["rErr"].array(library="np")[0]) if "rErr" in tree.keys() else None

        return {"best_fit": r, "uncertainty": r_err}

    def _parse_goodness_of_fit(self, results_file: str) -> Dict[str, Any]:
        """results_file is the "Observed" step's output
        (higgsCombineObserved.GoodnessOfFit.mH120*.root — combine may append a
        seed suffix); the "Toys" step's output (higgsCombineToys.GoodnessOfFit.
        mH120*.root, same directory) is read alongside it to compute a real
        p-value — matching Run2's two-file CollectGoodnessOfFit comparison,
        not a single-file stub."""
        with uproot.open(results_file) as f:
            observed = float(f["limit"]["limit"].array(library="np")[0])

        toys_matches = sorted(Path(results_file).parent.glob("higgsCombineToys.GoodnessOfFit.mH120*.root"))
        toys = None
        p_value = None
        if toys_matches:
            with uproot.open(str(toys_matches[0])) as f_toys:
                toys = f_toys["limit"]["limit"].array(library="np")
            p_value = float(np.mean(toys >= observed))

        return {"observed": observed, "toys": toys, "p_value": p_value}

    def _parse_impacts(self, results_file: str) -> Dict[str, Any]:
        with open(results_file) as f:
            data = json.load(f)
        impacts = {p["name"]: p.get("impact_r", 0.0) for p in data.get("params", [])}
        return {"impacts": impacts}
