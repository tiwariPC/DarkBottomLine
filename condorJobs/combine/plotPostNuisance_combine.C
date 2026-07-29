// Ported from Run2's bbDMlimitmodelrateParam_oneRP/plotPostNuisance_combine.C
// — CRonly pulls mode reads the fit_b RooFitResult directly (NOT via
// diffNuisances.py: with SR masked there is no S+B fit to compare against,
// only fit_b exists) and draws post-fit central value +/- post-fit
// uncertainty against a pre-fit-uncertainty reference band. This is a
// DIFFERENT tool from PlotPulls.C (used by the other 3 pulls modes, which do
// go through diffNuisances.py) — Run2's real pulls_oneRP.sh only calls this
// macro for the CRonly mode, verified by reading the full script.
//
// Only change from Run2's version: the hardcoded 2016/2017/2018/run2
// filename-substring lumi lookup is replaced with an explicit lumi argument
// (same adaptation as PlotPulls.C), and the stray file_h1_ref.root dump
// (Run2's own debug leftover, unrelated to the actual plot output) is
// dropped.

void plotPostNuisance_combine(TString mlfit="fitDiagnostics_none.root", TString outdir="",
                                TString postfix="CRonly", TString lumiText="",
                                double MaxUncertainty = 1.0, double MaxValue = 10.0,
                                bool skipmu = true) {

  TFile *File = TFile::Open(mlfit, "READ");
  RooFitResult *fit_b = (RooFitResult*)File->Get("fit_b");
  if(!fit_b) return;
  RooArgList parlist = fit_b->floatParsFinal();

  std::vector<TString> nuisance;
  std::vector<Double_t> central;
  std::vector<Double_t> uncert;
  std::vector<TString> nuisance_all;
  std::vector<Double_t> central_all;
  std::vector<Double_t> uncert_all;

  cout << "-------------------------------------------------------------------------------" << endl;
  cout << "Index :::               Nuisance name              :       Pull +/- uncertainty" << endl;
  cout << "-------------------------------------------------------------------------------" << endl;
  for(int i=0; i<parlist.getSize(); i++) {
    TString name_tstr = parlist[i].GetName();
    if( skipmu && name_tstr == "r" ) continue;

    central_all.push_back( ((RooRealVar&)parlist[i]).getVal() );
    uncert_all.push_back( ((RooRealVar&)parlist[i]).getError() );
    nuisance_all.push_back( parlist[i].GetName() );

    if(((RooRealVar&)parlist[i]).getVal() < 10){
        central.push_back( ((RooRealVar&)parlist[i]).getVal() );
        uncert.push_back( ((RooRealVar&)parlist[i]).getError() );
        nuisance.push_back( parlist[i].GetName() );
    }
  }
  cout << "-------------------------------------------------------------------------------" << endl;

  TCanvas *c = new TCanvas("c", "c", 1400, 900);
  c->SetBottomMargin(0.35);

  int nbins = central.size()+2;
  TH1D* h1        =   new TH1D("h1",      "h1",       nbins, -0.5, nbins-0.5);
  TH1D* h1_ref    =   new TH1D("h1_ref",  "h1_ref",   nbins, -0.5, nbins-0.5);

  for(int i=0; i<nbins-2; i++) {
    h1->SetBinContent(i+2, central[i]);
    h1->SetBinError(i+2, uncert[i]);
    h1_ref->GetXaxis()->SetBinLabel(i+2, nuisance[i]);
    h1_ref->SetBinContent(i+2, central[i]);
    h1_ref->SetBinError(i+2, 1.);
  }

  TPaveText *pt = new TPaveText(0.0877181,0.9,0.9580537,0.96,"brNDC");
  pt->SetBorderSize(0);
  pt->SetTextAlign(12);
  pt->SetFillStyle(0);
  pt->SetTextFont(52);

  double cmstextSize = 0.07;
  double preliminarytextfize = cmstextSize * 0.7;
  double lumitextsize = cmstextSize *0.7;
  pt->SetTextSize(cmstextSize);
  pt->AddText(0.01,0.57,"#font[61]{CMS}");

  TPaveText *pt1 = new TPaveText(0.0877181,0.905,0.9580537,0.96,"brNDC");
  pt1->SetBorderSize(0);
  pt1->SetTextAlign(12);
  pt1->SetFillStyle(0);
  pt1->SetTextFont(52);
  pt1->SetTextSize(preliminarytextfize);
  pt1->AddText(0.125,0.4,"Internal");

  TPaveText *pt2 = new TPaveText(0.0877181,0.9,0.8280537,0.96,"brNDC");
  pt2->SetBorderSize(0);
  pt2->SetTextAlign(12);
  pt2->SetFillStyle(0);
  pt2->SetTextFont(42);
  pt2->SetTextSize(lumitextsize);
  if (lumiText.Length() > 0) {
    pt2->AddText(0.81, 0.5, lumiText);
  }

  TPaveText *pt3 = new TPaveText(0.0377181,0.85,0.9580537,0.88,"brNDC");
  pt3->SetBorderSize(0);
  pt3->SetTextAlign(12);
  pt3->SetFillStyle(0);
  pt3->SetTextFont(42);
  pt3->SetTextSize(lumitextsize);
  pt3->AddText(0.1,0.4, TString("CRonly Fit"));

  h1_ref->LabelsOption("v");
  h1_ref->SetStats(0);
  h1_ref->SetTitle("");
  h1_ref->SetYTitle("Pull");
  h1_ref->SetFillColor(18);
  h1_ref->SetMinimum(-3.0);
  h1_ref->SetMaximum(3.0);
  h1_ref->GetYaxis()->SetNdivisions(3, false);
  h1->SetLineColor(kBlack);

  TH1D* h1_zero = new TH1D("h1_zero", "h1_zero", 1, -0.5, nbins-0.5);
  h1_zero->SetBinContent(1,0);
  h1_zero->SetLineStyle(2);

  h1_ref->Draw("E2");
  h1_zero->Draw("histo same");
  h1->Draw("same E");
  h1->SetLineColor(2);
  h1->SetLineWidth(2);
  h1->SetMarkerStyle(20);
  h1->SetMarkerSize(1.5);
  pt->Draw();
  pt1->Draw();
  pt2->Draw();
  pt3->Draw();

  int nuisanceRange = 90;
  int numberOfNuisance = nuisance_all.size();
  int numberOfCanvas = (numberOfNuisance/nuisanceRange) +1;

  for (int i =1 ; i < numberOfCanvas+1 ; i++)
      {
      string postfix_2= to_string(i);
      int last = nuisanceRange * i;
      int start = nuisanceRange * ( i -1 );
      if (i==1) {
              h1_ref->SetAxisRange(start, last, "X");
              }
      if (i!=1 && i < numberOfCanvas) {
              h1_ref->SetAxisRange(start+1, last, "X");
              }
      if (i == numberOfCanvas){
              h1_ref->SetAxisRange(start+1, numberOfNuisance+1, "X");
              }
      c->Update();
      c->Modified();

      TString postfix_two(postfix_2.c_str());
      TString pdfOutputFilePath = TString::Format("%s/pulls_%s_%s.pdf", outdir.Data(), postfix.Data(), postfix_two.Data());
      c->SaveAs(pdfOutputFilePath.Data());

      TString pngOutputFilePath = TString::Format("%s/pulls_%s_%s.png", outdir.Data(), postfix.Data(), postfix_two.Data());
      c->SaveAs(pngOutputFilePath.Data());

      TString rootOutputFilePath = TString::Format("%s/pulls_%s_%s.root", outdir.Data(), postfix.Data(), postfix_two.Data());
      c->SaveAs(rootOutputFilePath.Data());
  }
}
