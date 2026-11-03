"""Helpers for novel (engineered) CAST systems.

Predicts where a CAST should insert for a given spacer, using the spacing
measured from our profiling data rather than observed sequencing reads.
For the same quantities measured from real insertions, see
TsvMapHelperFunctions.characterize_insertion_from_spacer.
"""

import pandas as pd

# Insertion point measured from the end of the PAM, and the +/- window used
# to pull target-site sequence context around it.
INSERTION_DISTANCE = 71
TARGET_SITE_WINDOW = 10


def predict_insertion_from_spacer(guideRNA, genome_sequence):
    #from spacer find insertions that occur at the 'correct' distance:
    from . import postprocessHelpers as hp

    distance = INSERTION_DISTANCE
    window = TARGET_SITE_WINDOW

    grna = hp.map_guide_to_genome(guideRNA,genome_sequence)
    if grna[0]['strand'] == '+':
        predicted_pos = grna[0]['end']+3+distance
    else:
        predicted_pos = grna[0]['start']-3-distance

    df = pd.DataFrame(columns=["PAM","protospacer","gRNA_strand","intervening","insertion_length","target_site",
                               "PAM_start","PAM_end","proto_start","proto_end","aln_score",
                               "intervene_start","intervene_end","ts_start","ts_end"])

    df['proto_start']= [ grna[0]['start'] ]
    df['proto_end']= [ grna[0]['end'] ]
    df['protospacer']= [ grna[0]['sequence'] ]
    if grna[0]['strand'] == '+':
        df['PAM_start'] = [ grna[0]['end'] ]
        df['PAM_end']= [ grna[0]['end']+3 ]
        df['PAM']= [ genome_sequence[ grna[0]['end'] : grna[0]['end']+3 ] ]
        df['intervene_start']= [ grna[0]['end']+3 ]
        df['intervene_end']= [ predicted_pos ]
        df['intervening']= [ genome_sequence[ grna[0]['end']+3 :  predicted_pos  ] ]
        df['target_site']= [ genome_sequence[predicted_pos-window:predicted_pos+window] ]
        df['insertion_length']= [ len(genome_sequence[ grna[0]['end']+3 :  predicted_pos  ]) ]

    else:
        df['PAM_start'] = [ grna[0]['start']-3 ]
        df['PAM_end']= [ grna[0]['start'] ]
        df['PAM']= [ genome_sequence[grna[0]['start']-3:grna[0]['start']]  ] #always list top strand
        df['intervene_start']= [ predicted_pos ]
        df['intervene_end']= [ grna[0]['start']-3 ]
        df['intervening']= [ genome_sequence[predicted_pos:grna[0]['start']-3] ]
        df['target_site']= [ genome_sequence[predicted_pos-window:predicted_pos+window]  ]
        df['insertion_length']=[ len(genome_sequence[predicted_pos:grna[0]['start']-3] ) ]

    df['ts_start']= [ predicted_pos-window ]
    df['ts_end']= [ predicted_pos+window ]
    df['gRNA_strand']= [ grna[0]['strand'] ]
    return df
