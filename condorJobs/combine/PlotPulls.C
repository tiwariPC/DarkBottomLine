// Ported from Run2's bbDMlimitmodelrateParam_oneRP/PlotPulls.C — renders the
// "nuisances" TCanvas that diffNuisances.py -g writes (a prefit_nuisancs
// TH1F pull histogram + legend) into paginated CMS-style plots, matching
// Run2's real output format exactly (one .pdf/.png/.root triplet per ~89
// nuisances). The only change from Run2's version: the hardcoded
// 2016/2017/2018/run2 filename-substring lumi lookup is replaced with an
// explicit lumi/year argument (Run3's eras aren't known at macro-write time
// the way Run2's fixed 3-era list was), passed in by combine_tools.py from
// combine.yaml's own eras[].year_config lumi value.
#include <string>

void PlotPulls(TString filename="pulls_none.root", TString outdir="",
                TString postfix_="", TString lumiText="", TString dataLabel="Data"){

  TString plotdir = outdir;
    TFile file(filename,"READ");
    TCanvas *c = (TCanvas*)file.Get("nuisances");
    c->ls(); //check inside the c canvas
    c->Size(1250,400);
    c->SetBottomMargin(0.35);
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TH1F *h1 = (TH1F*)c->GetPrimitive("prefit_nuisancs");
    h1->LabelsOption("v");
    h1->SetMinimum(-3.00);
    h1->SetMaximum(3.0);
    int numberOfNuisance = h1->GetXaxis()->GetNbins();
    TLegend leg1 = TLegend(0.6, 0.74, 0.89, 0.89);
    TLegend *leg2 = (TLegend*)(c->GetPrimitive("TPave"));
    leg1.Copy(*leg2);

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
    pt3->AddText(0.1,0.4, dataLabel+" Fit");

    h1->Draw("same");
    leg2->Draw();
    pt->Draw();
    pt1->Draw();
    pt2->Draw();
    pt3->Draw();

    int nuisanceRange = 89;
    int numberOfCanvas = (numberOfNuisance/nuisanceRange) +1;
    cout << "total number of canvas " << numberOfCanvas << endl;
    for (int i =1 ; i < numberOfCanvas+1 ; i++)
	{
	string postfix= to_string(i);
	string prefix= to_string(i-1);
	int last = nuisanceRange * i;
	int start = nuisanceRange * ( i -1 );
	cout << "start  "  << start << "  last  " << last << endl;
	if (i==1) {
		h1->SetAxisRange(start, last, "X");
		}
	if (i!=1 && i < numberOfCanvas) {
		h1->SetAxisRange(start+1, last, "X");
		}
	if (i == numberOfCanvas){
		h1->SetAxisRange(start+1, numberOfNuisance+1, "X");
		}
	c->Update();
  TString lastName = gSystem->BaseName(filename);
	c->SaveAs(plotdir+lastName.ReplaceAll(".root","_"+postfix+"_.pdf").ReplaceAll("_"+prefix+"_","").ReplaceAll("/","_"));
	c->SaveAs(plotdir+lastName.ReplaceAll(".pdf",".png").ReplaceAll("/","_"));
	c->SaveAs(plotdir+lastName.ReplaceAll(".png",".root").ReplaceAll("/","_"));

	}
}
