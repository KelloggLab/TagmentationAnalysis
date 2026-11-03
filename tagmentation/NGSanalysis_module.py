from Bio import SeqIO
from Bio.Seq import Seq
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os

def fasta_to_string(fasta_path):
    return str(next(SeqIO.parse(fasta_path, "fasta")).seq)

def trim_reads_by_primer(fastq_filepath, primer):
    total_reads = []
    
    trimmed_reads = []

    primer = primer.upper()

    primer_len = len(primer)
    for rec in SeqIO.parse(fastq_filepath, "fastq"):
        seq = str(rec.seq)
        total_reads.append(seq)
        primer_idx = seq.find(primer)
        if primer_idx != -1:
            trimmed_seq = seq[primer_idx + primer_len:]
            trimmed_qual = rec.letter_annotations["phred_quality"][primer_idx + primer_len:]
            trimmed_reads.append((rec.id, trimmed_seq, trimmed_qual))
    
    if len(trimmed_reads) > 0:
        print(f"Total reads before trimming: {len(total_reads)}")
        return trimmed_reads

    else:
        print("Could not trim reads -- double check your primer sequence and offset.")


def write_trimmed_fastq(trimmed_reads, output_path):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_path, "w") as f:
        for read_id, seq, qual in trimmed_reads:
            qual_str = ''.join([chr(q + 33) for q in qual])
            f.write(f"@{read_id}\n{seq}\n+\n{qual_str}\n")

import subprocess
import re

CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')

def aligned_ref_length(cigar):
    """Number of reference bases consumed by alignment"""
    length = 0
    for n, op in CIGAR_RE.findall(cigar):
        if op in "MDN=X":
            length += int(n)
    return length

def extract_insertion_sites(samtools_bin,bam_file, isforward, tn5_offset=5):
    result = subprocess.run(
        [str(Path(samtools_bin) / "samtools"), "view", str(bam_file)],
        capture_output=True,
        text=True,
        check=True
    )

    positions = []

    for line in result.stdout.splitlines():
        if not line:
            continue

        fields = line.split("\t")
        #print(line)

        flag   = int(fields[1])
        pos1   = int(fields[3])      # 1-based leftmost position
        cigar  = fields[5]

        start0 = pos1 - 1             # convert to 0-based
        ref_len = aligned_ref_length(cigar)

        is_reverse = flag & 16

        if not is_reverse and isforward:
            # forward strand: insertion at left edge
            ins = start0 + tn5_offset
            positions.append(ins)
        elif is_reverse and not isforward:
            # reverse strand: insertion at right edge
            ins = start0 + ref_len - 1 - tn5_offset
            positions.append(ins)

        

    return positions

	
def sgRNA_finder(sgRNA, genome, PAM_len, natural_system):
    sgRNA = sgRNA.upper()
    genome = genome.upper()
    
    if natural_system:
        if genome.find(sgRNA) != -1:
            return genome.find(sgRNA), genome.find(sgRNA), genome.find(sgRNA) - PAM_len, False
        elif genome.find(str(Seq(sgRNA).reverse_complement())) != -1:
            return genome.find(str(Seq(sgRNA).reverse_complement())), genome.find(str(Seq(sgRNA).reverse_complement()))  + len(sgRNA), genome.find(str(Seq(sgRNA).reverse_complement())) + len(sgRNA) + PAM_len, True
        else:
            print("sgRNA Sequence Not Found in Genome")
    
    else:
        if genome.find(sgRNA) != -1:
            return genome.find(sgRNA), genome.find(sgRNA) + len(sgRNA), genome.find(sgRNA) + len(sgRNA) + PAM_len, False
        elif genome.find(str(Seq(sgRNA).reverse_complement())) != -1:
            return genome.find(str(Seq(sgRNA).reverse_complement())), genome.find(str(Seq(sgRNA).reverse_complement())), genome.find(str(Seq(sgRNA).reverse_complement())) - PAM_len, True
        else:
            print("sgRNA Sequence Not Found in Genome")


def save_to_temp(uploadedFile,tmpdir):
#	with tempfile.NamedTemporaryFile(dir=tmpdir.name,delete=False, suffix=".fasta") as tmp_file:
#	directory = os.path.dirname(tmpdir)
	if not os.path.exists(tmpdir):
		os.makedirs(tmpdir)
	with tempfile.NamedTemporaryFile(dir=tmpdir,delete=False, suffix=".fasta") as tmp_file:
		tmp_file.write(uploadedFile.getvalue())
		temp_path = tmp_file.name  # Full path to the saved file
		print("temporary file: " + temp_path)
	return temp_path

def integration_finder(insertion_idx_list, ccdb_gene, ccdb_start_site, PAM_start_site, PAM_end_site, natural_system, is_rc):
    dist_list = []

    if natural_system:
        if is_rc:
            def dist_calc(idx): return (PAM_start_site - idx)

        else:
            def dist_calc(idx): return (idx - PAM_start_site)
    
    else:
        if is_rc:
            def dist_calc(idx): return (PAM_end_site - idx)

        else:
            def dist_calc(idx): return (idx - PAM_end_site)

    for idx in insertion_idx_list:
        if ccdb_start_site + len(ccdb_gene) >= idx >= ccdb_start_site:
            modified_idx = dist_calc(idx)
            dist_list.append(modified_idx)

    return dist_list

#the main function
from pathlib import Path
import subprocess

def process_reads(fastq, primer, first_ten_in_donor, refgenome, bbmap_bin, samtools_bin, path_to_output_dir):
    path_to_output_dir = Path(path_to_output_dir)
    path_to_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trim reads by primer
    trimmed_reads = trim_reads_by_primer(fastq, primer)

    # 2. Remove donor reads
    donor_removed_reads = [read for read in trimmed_reads if not read[1].startswith(first_ten_in_donor)]
    print(f"Total reads after primer trimming: {len(trimmed_reads)}")
    print(f"Total reads after donor removal: {len(donor_removed_reads)}")

    # 3. Write output FASTQ
    trimmed_fastq = path_to_output_dir / "trimmed.fastq"
    write_trimmed_fastq(donor_removed_reads, trimmed_fastq)

    # 4. Index + map
    bbmap_sh = str(Path(bbmap_bin) / "bbmap.sh")
    subprocess.run([bbmap_sh, f"ref={refgenome}", "-Xmx20g", "ztd=1"], check=True)

    mapped_bam = trimmed_fastq.with_suffix(".bam")
    subprocess.run([
        bbmap_sh,
        f"in={str(trimmed_fastq)}",
        f"outm={str(mapped_bam)}",
        "-Xmx20g", "ztd=1",
        "minid=0.9", "ambig=toss", "local=t", "slow=t", "tipsearch=200",
    ], check=True)

    # 5. Optional: quick sanity check that BAM has alignments
    samtools = str(Path(samtools_bin) / "samtools")
    n_lines = subprocess.check_output([samtools, "view", str(mapped_bam)]).splitlines()
    print(f"Mapped alignments: {len(n_lines)}")

    # 6. Extract insertion sites (prefer doing strand logic inside)
    # If your extract_insertion_sites still takes strand="f"/"r", keep it:
    forward_insertions = extract_insertion_sites(samtools_bin,mapped_bam,isforward=True,tn5_offset=5)
    reverse_insertions = extract_insertion_sites(samtools_bin,mapped_bam,isforward=False,tn5_offset=5)

    return [forward_insertions, reverse_insertions]


def plot_insertion_histogram(
    forward_positions, reverse_positions,
    region_start=0, region_end=None,
    bins=100, title="Insertion Sites",
    highlight_regions=None,
    filename=None
):
    if region_end is None:
        region_end = max(forward_positions + reverse_positions)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        forward_positions, bins=bins, range=(region_start, region_end),
        alpha=0.5, label='Top Strand', color='black'
    )
    ax.hist(
        reverse_positions, bins=bins, range=(region_start, region_end),
        alpha=0.5, label='Bottom Strand', color='grey'
    )

    if highlight_regions:
        ymin, ymax = ax.get_ylim()
        y = ymin - 0.01 * (ymax - ymin)
        for region in highlight_regions:
            if len(region) == 3:
                x_start, x_end, color = region
            else:
                x_start, x_end = region
                color = 'red'
            ax.hlines(y=y, xmin=x_start, xmax=x_end, colors=color, linewidth=5)

    ax.set_xlabel("Genomic Position")
    ax.set_ylabel("Insertion Count")
    ax.set_title(title)
    ax.set_xlim(region_start, region_end)
    ax.legend()
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300)

    return fig, ax


def plot_integration_profile(data, title="Integration Distance From PAM", xlabel="Distance From PAM (bp)", ylabel="Count", 
                   xlabel_fontsize=12, ylabel_fontsize=12, title_fontsize=14, 
                   hist_color='grey', edge_color='black', linewidth=5, dpi=300, xlim=(20,100), filename=None):
    
    plt.clf()
    plt.hist(data, edgecolor=edge_color, color=hist_color, linewidth=linewidth)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    
    if xlim:
        plt.xlim(xlim)
    
    plt.gcf().set_dpi(dpi)
    
    plt.tick_params(axis='both', labelsize=10)

    if filename:
        plt.savefig(filename, dpi=dpi)
        plt.close()
    else:
        plt.show()
    return plt
