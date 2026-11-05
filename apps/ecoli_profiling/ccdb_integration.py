"""On-target scoring against the ccdB landing site.

Takes insertion positions and the ccdB gene's coordinates and reports how far
each insertion fell from the PAM. The ccdB target makes this specific to the
E. coli profiling assay, which is why it lives here and not in `core/`.
"""

from __future__ import annotations


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
