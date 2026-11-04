"""Kellogg Lab functions shared across the assays in `apps/`.

CAST integration profiling in E. coli, UdiTaS in human cells and ChIP-seq all
draw on the same sequence, interval and table handling, and it lives here.

`tagmentation` remains importable as an alias, so notebooks written against
the old name keep working.
"""
