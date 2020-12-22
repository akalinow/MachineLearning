## ML model for estimation of the muon transverse momentum in the CMS detector

This package is used fot training a deep NN for estimation of the momentum of muons in the
[CMS](http://cms.cern) detector. The momentum is estimated at the so called
[Level 1 of the trigger system](https://cms.cern/detector/triggering-and-data-acquisition).
At this point of data processing there is only about 1 microsecond available for calculations - therefore only
very simple and fast algorithms can be used. The data is processed on electronic boards dedigned, and build by the
[Warsaw CMS group](http://cms.fuw.edu.pl/?page_id=1200). Since the algorithm has to very simple a regression task
is formulated as categorisation one: doeas a muons belong to one of 54 bins of transverse momenta?

The current algorithm is a simple algoritmh using so called [naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
classificator. **This package is an attempt to use NN for the same task**

The ML is made using the [OMTF_NN](OMTF_NN.ipynb) notebook. This notebook can be run with
a [akalinow/tensorflow-gpu](https://hub.docker.com/r/akalinow/tensorflow-gpu) Docker container.
The training is run on Core i9-9900K CPU with NVIDIA GTX 2070 SUPER GPU. Please see the
notebook setcion for some example plots.

**We are looking for students to continue and extend this study**

If you are interested please contact [Artur.Kalinowski@fuw.edu.pl](Artur.Kalinowski@fuw.edu.pl).