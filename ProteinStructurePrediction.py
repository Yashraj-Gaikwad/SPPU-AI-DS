'''
Assignment: Protein Structure Prediction. Task: Predict the 3D structure of a given protein sequence using
homology modeling or threading techniques. Deliverable: A report presenting the predicted protein structure,
along with an analysis of its potential functions and interactions

Predicting the 3D structure of a protein is a multi-step process that involves retrieving the protein sequence, modeling its structure, and validating the predicted model. Here’s a detailed explanation of each step:

     Step 1: Retrieve Protein Sequence from UniProt  

1.   Access UniProt:  
   - Visit the UniProt website (https://www.uniprot.org/), which is a comprehensive resource for protein sequences and functional information.

2.   Search for the Target Protein:  
   - Use the search bar to enter either the name or the accession number of the protein you are interested in (e.g., "xyz protein"). This will help you locate the specific entry for that protein.

3.   Copy the Protein Sequence:  
   - Once you find the protein entry, navigate to the section containing the protein sequence. Copy this sequence in FASTA format, which is a standard format used for representing nucleotide or peptide sequences.

     Step 2: Predict the 3D Structure Using SWISS-MODEL  

1.   Access SWISS-MODEL:  
   - Go to the SWISS-MODEL website (https://swissmodel.expasy.org/) where you can perform homology modeling of proteins.

2.   Paste the FASTA Sequence:  
   - In the modeling tool on SWISS-MODEL, paste the copied FASTA sequence of your target protein. This is necessary for initiating the modeling process.

3.   Select Templates:  
   - The tool will automatically search for suitable templates based on sequence similarity to known structures. Review these templates and select one with the highest alignment score or maximum sequence identity to your target protein, as this will yield a more accurate model.

4.   Run the Modeling Process:  
   - Start the modeling process, which aligns your target sequence with the selected template and generates a predicted 3D structure in PDB (Protein Data Bank) format. This file format is widely used for representing 3D structures of biological macromolecules.

     Step 3: Validate the Predicted Structure  

1.   Use SAVES Server:  
   - Visit the SAVES Server (http://services.mbi.ucla.edu/SAVES/) for structural validation of your predicted model.

2.   Upload the Predicted PDB File:  
   - Upload your generated PDB file to the SAVES server, which will analyze its structural quality.

3.   Check ERRAT Score:  
   - The ERRAT tool evaluates non-ideal geometry in your structure. A high ERRAT score indicates fewer structural issues and better quality.

4.   Use PROCHECK:  
   - Utilize PROCHECK to validate geometric parameters of your protein structure. It provides insights into:
   -   Ramachandran Plot:   This plot assesses dihedral angles (phi and psi) of amino acids, indicating whether they fall within allowed regions.
   -   Bond Lengths and Angles:   PROCHECK evaluates bond lengths and angles to ensure they are consistent with known values, further confirming structural integrity.

By following these steps, you can effectively predict and validate the 3D structure of a protein, contributing valuable insights into its function and interactions within biological systems.


1) P0AFF6 · NUSA_ECOLI

>sp|P0AFF6|NUSA_ECOLI Transcription termination/antitermination protein NusA OS=Escherichia coli (strain K12) OX=83333 GN=nusA PE=1 SV=1
MNKEILAVVEAVSNEKALPREKIFEALESALATATKKKYEQEIDVRVQIDRKSGDFDTFR
RWLVVDEVTQPTKEITLEAARYEDESLNLGDYVEDQIESVTFDRITTQTAKQVIVQKVRE
AERAMVVDQFREHEGEIITGVVKKVNRDNISLDLGNNAEAVILREDMLPRENFRPGDRVR
GVLYSVRPEARGAQLFVTRSKPEMLIELFRIEVPEIGEEVIEIKAAARDPGSRAKIAVKT
NDKRIDPVGACVGMRGARVQAVSTELGGERIDIVLWDDNPAQFVINAMAPADVASIVVDE
DKHTMDIAVEAGNLAQAIGRNGQNVRLASQLSGWELNVMTVDDLQAKHQAEAHAAIDTFT
KYLDIDEDFATVLVEEGFSTLEELAYVPMKELLEIEGLDEPTVEALRERAKNALATIAQA
QEESLGDNKPADDLLNLEGVDRDLAFKLAARGVCTLEDLAEQGIDDLADIEGLTDEKAGA
LIMAARNICWFGDEA

2) https://swissmodel.expasy.org/interactive/7PkgFV/models/

3) https://saves.mbi.ucla.edu/?job=83895




Viva Questions
1) 

'''

