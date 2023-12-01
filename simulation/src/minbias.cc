/*

  Authors: 
  Matias Mantinan
  Anthony Badea
  Karri DiPetrillo

  Description:
  Script to generate pythia samples

*/

// c, pythia, root includes
#include <iostream>
#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"

// declare pythia8 namespace
using namespace Pythia8;

// print a message to the console
void msg(string m){
  printf("\r%s",m.c_str());                               
  std::cout << std::endl;
}

// progress bar for the people taken from alex tuna and ann wang
void pbftp(double time_diff, int nprocessed, int ntotal){
  if(nprocessed%10 == 0){
    double rate      = (double)(nprocessed+1)/time_diff;
    std::cout << "\r > " << nprocessed << " / " << ntotal
	      << " | "   << std::fixed << std::setprecision(1) << 100*(double)(nprocessed)/(double)(ntotal) << "%"
	      << " | "   << std::fixed << std::setprecision(1) << rate << "Hz"
	      << " | "   << std::fixed << std::setprecision(1) << time_diff/60 << "m elapsed"
	      << " | "   << std::fixed << std::setprecision(1) << (double)(ntotal-nprocessed)/(rate*60) << "m remaining"
	      << std::flush;
  }
  // add new line at end of events
  if (nprocessed+1 == ntotal){
    msg("");
  }
}

// anthony write recurisive to walk down higgs chain to get dark quarks
const int getFinalParent(auto event, int iP, int pdgid){
  int d1 = event[iP].daughter1();
  int d2 = event[iP].daughter2();
  int id = event[iP].id();

  // check if daughter is same id
  if( d1 != d2 ){
    return iP;
  }
  else{
    // if daughter id is the same as pdgid then keep stepping
    if( event[d1].id() == pdgid ){
      return getFinalParent(event, d1, pdgid);
    }
    // if daughter id is different then stop
    else{
      return iP;
    }
  }

  return -1;
}

int main(int argc, char* argv[]) {

  // check user is inputting correct parameters
  if(argc != 4){
    std::cout << "Usage: ./higgsPortal.exe <pythiaCard> <outFileName> <maxEvents>" << std::endl;
    return 1;
  }
  // read in user parameters
  std::string pythiaCard = argv[1];
  std::string outFileName = argv[2];
  int maxEvents = std::stoi(argv[3]);

  // Generator.
  Pythia pythia;

  // Shorthand for the event record in pythia.
  Event& event = pythia.event;

  // Read in commands from external file.
  pythia.readFile(pythiaCard.c_str());

  // Initialize.
  pythia.init();

  // Create root file and tree
  TFile* f = new TFile(outFileName.c_str(),"RECREATE");
  TTree* t = new TTree("t","t");

  // initialize branches variables
  int nParticles;
  std::vector<int> *id = 0;
  std::vector<int> *status = 0;
  std::vector<double> *mass = 0;
  std::vector<double> *pt = 0;
  std::vector<double> *eta = 0;
  std::vector<double> *phi = 0;
  std::vector<double> *tau = 0;
  std::vector<double> *xDec = 0;
  std::vector<double> *yDec = 0;
  std::vector<double> *zDec = 0;
  std::vector<double> *tDec = 0;

  // initialize branches
  t->Branch("nParticles", &nParticles);
  t->Branch("id", &id);
  t->Branch("status", &status);
  t->Branch("mass", &mass);
  t->Branch("pt", &pt);
  t->Branch("eta", &eta);
  t->Branch("phi", &phi);
  t->Branch("tau",&tau);
  t->Branch("xDec",&xDec);
  t->Branch("yDec",&yDec);
  t->Branch("zDec",&zDec);
  t->Branch("tDec",&tDec);

  // time keeper for progress bar
  std::chrono::time_point<std::chrono::system_clock> time_start;
  std::chrono::duration<double> elapsed_seconds;
  time_start = std::chrono::system_clock::now();

  for (int iE = 0; iE < maxEvents; ++iE) {

    if(!pythia.next()) continue;

    // progress bar
    elapsed_seconds = (std::chrono::system_clock::now() - time_start);
    pbftp(elapsed_seconds.count(), iE, maxEvents);

    // clear for new event
    id->clear();
    status->clear();
    mass->clear();
    pt->clear();
    eta->clear();
    phi->clear();
    tau->clear();
    xDec->clear();
    yDec->clear();
    zDec->clear();
    tDec->clear();
    
    // get event level information
    nParticles = event.size();
    
    // helpful flags
    bool finalHiggs = false;
    
    // loop over the particles. available properties listed here https://pythia.org/latest-manual/ParticleProperties.html
    for(int iP=0; iP<nParticles; iP++){
      
      // locate the higgs
      // if(pythia.event[iP].id() == 25){
	
      // 	// identify final higgs in the chain
      // 	const int iHiggs =getFinalParent(pythia.event, iP, 25);
      // 	// std::cout<<iHiggs<<std::endl;
	
      // 	// get higgs daughter index ranges
      // 	int d1 = pythia.event[iHiggs].daughter1();
      // 	int d2 = pythia.event[iHiggs].daughter2();
	
      // 	// identify dark quarks
      // 	if(abs(pythia.event[d1].id()) == 4900101){

      // 	  // loop over the dark quarks
      // 	  for (int iQ = d1; iQ <= d2; ++iQ) {
	    
      // 	    // identify final dark quark in the chain
      //       const int iDQ = getFinalParent(pythia.event, iQ, pythia.event[iQ].id());
      // 	    // std::cout<< iQ << ", " << iDQ << ", " << pythia.event[iDQ].id() << ", " << pythia.event[iDQ].daughter1() << ", " << pythia.event[iDQ].daughter2() << std::endl;

      // 	    // loop over the dark quarks daughters
      // 	    for (int iF = pythia.event[iDQ].daughter1(); iF <= pythia.event[iDQ].daughter2(); ++iF) {
      // 	      std::cout<< "   " << iF << ", " << pythia.event[iF].id() << ", " << pythia.event[iF].daughter1() << ", " << pythia.event[iF].daughter2() << std::endl;
      // 	    }
      // 	  }
      // 	}
      // }

      // save particle information
      id->push_back(pythia.event[iP].id());
      status->push_back(pythia.event[iP].status());
      mass->push_back(pythia.event[iP].m());
      pt->push_back(pythia.event[iP].pT());
      eta->push_back(pythia.event[iP].eta());
      phi->push_back(pythia.event[iP].phi());
      tau->push_back(pythia.event[iP].tau());
      xDec->push_back(pythia.event[iP].xDec());
      yDec->push_back(pythia.event[iP].yDec());
      zDec->push_back(pythia.event[iP].zDec());
      tDec->push_back(pythia.event[iP].tDec());
   }

    // fill tree on each particle loop
    t->Fill();
  } 

  // write and cleanup
  t->Write();  
  delete t;

  // close file and cleanup
  f->Close();
  delete f;

  return 0;
}
