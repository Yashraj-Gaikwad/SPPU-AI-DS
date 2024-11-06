'''
Assignment: DNA Sequence Analysis. Task: Analyze a given DNA sequence and perform basic sequence
manipulation, including finding motifs, calculating GC content, and identifying coding regions. Deliverable: A
report summarizing the analysis results and any insights gained from the sequence

DNA Sequence = 5'-ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGC-3'
Motif to find = ATG

Viva Questions
1) Explain DNA
2) Explain Motifs, GC content, ORFs

'''

# 1) Find motif
def find_motif(dna_sequence, motif):
   
    # list of indexs where motif is found
    positions = []

    # doesnt let it overflow
    for i in range(len(dna_sequence) - len(motif) + 1):
        
        # compare substring
        if dna_sequence[i:i + len(motif)] == motif:
            # add to list
            positions.append(i)
            
    return positions

# 2) calculate gc content
def calculate_gc_content(dna_sequence):
    
    g_count = dna_sequence.count('G')
    
    c_count = dna_sequence.count('C')
    
    gc_content = (g_count + c_count) / len(dna_sequence) * 100
    
    return gc_content

# 3) Find orfs - Open Reading Frames
def find_orfs(dna_sequence):
    orfs = []
    start_codon = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]

    # start where i can euqal to first three nucleotides
    for i in range(len(dna_sequence) - 2):
        
        # extract a condon of length 3
        codon = dna_sequence[i:i+3]
        
        if codon == start_codon:
               
            # start checking after condon
            # loop increments by 3 to check condons
            for j in range(i+3, len(dna_sequence) - 2, 3):
                   
                # extract condon
                next_codon = dna_sequence[j:j+3]
                
                # compare
                if next_codon in stop_codons:

                    orfs.append(dna_sequence[i:j+3])
                    break
    return orfs

def main():

    # Input: DNA sequence
    dna_sequence = input("Enter the DNA sequence: ").upper()
    
    # Finding Motif
    motif = input("Enter the motif to find: ").upper()
    motif_positions = find_motif(dna_sequence, motif)
    print(f"\nMotif {motif} found at positions: {motif_positions}")
    
    # Calculating GC Content
    gc_content = calculate_gc_content(dna_sequence)
    print(f"GC Content: {gc_content:.2f}%")
    
    # Identifying Coding Regions (ORFs)
    orfs = find_orfs(dna_sequence)
    print(f"Identified ORFs: {orfs}")

if __name__ == "__main__":
    main()
