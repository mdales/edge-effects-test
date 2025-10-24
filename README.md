Some initial experiments with edge effects for AOHs. In this repo I have

* aoh.py - a straight simple AOH implementation
* aohedge.py - a simple binary edge effect implementation that removes pixels from the edges for a given radius
* aohfractionaledge.py - allows between 0.0 and 1.0 of a pixel to be defined as an edge
* findedges.py - a script that shows which pixels are edges in a raster
* summarize_results.py - takes a directory of AOHs for species/edge size and generates a CSV of the results