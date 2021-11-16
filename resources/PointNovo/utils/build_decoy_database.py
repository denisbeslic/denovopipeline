from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def build_decoy(input_database_file_name, output_database_file_name):
    sequences = []
    with open(input_database_file_name, "r") as input_handle:
        record_iterator = SeqIO.parse(input_handle, "fasta")
        for record in record_iterator:
            reverse_seq = record.seq[::-1]
            name = record.name + "_DECOY"
            id = record.id + "_DECOY"
            decoy_record = SeqRecord(reverse_seq,
                                     id=id,
                                     name=name,
                                     description=record.description)
            sequences.append(record)
            sequences.append(decoy_record)

    with open(output_database_file_name, 'w') as output_handle:
        SeqIO.write(sequences, output_handle, 'fasta')


if __name__ == "__main__":
    input_file_name = '/home/rui/work/DeepNovo-pytorch/fasta_files/uniprot_sprot_human.fasta'
    output_file_name = '/home/rui/work/DeepNovo-pytorch/fasta_files/uniprot_sprot_human_with_decoy.fasta'
    build_decoy(input_file_name, output_file_name)
