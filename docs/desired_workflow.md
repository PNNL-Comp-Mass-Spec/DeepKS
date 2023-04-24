- `DeepKS.api.main`:
  -  Obtain novel predictions with the following options:
     - Input 
       - kinase/site sequences by typing on command line or specifying input file
         - TODO: Enable FASTA input
       - Create pairs by matching up or by a cartesian product
     - Output
       - Specify format
       - Include sequences or not
       - Include groups or not
       - Normalize scores; TODO: this should be removed in favor of --convert-raw-to-prob
     - Logging
       - There is a -v option, but it doesn't do anything; TODO: use logging module instead
     - Prediction options
       - Specify NN model
       - Specify GC model
       - Bypass GC or not
       - Convert raw scores to probabilites or not
       - TODO: Specify metric (default ROC)
       - TODO: Specify cutoff (default auto)
       - TODO: Specify max allowable FPR (default 0.1)
     - Misc
       - Dry run which validates all inputs and stops short of calling `MultiStageClassifier.predict`
       - Help facility

- `DeepKS.models.individual_classifiers`:
  - Training and Validation
    - Specify Device
    - Specify Train file
    - Specify Val file
    - Specify whether or not to save model
    - TODO: JSON-configurable hyperparameters
  - TODO: NO Testing (saved for MultiStageClassifier), NO ROC (saved for MultiStageClassifier)

- `DeepKS.models.multi_stage_classifier`:
  - TODO: ONLY defines the class `MultiStageClassifier`; remove main and parsing from this module

- TODO: New module `DeepKS.models.train_group_classifier`:
  - Specify whether or not to save GC
  - TBD

- `DeepKS.models.DeepKS_evaluation`:
  - Test Performance for each Group
    - Specify Test file
    - Specify whether or not to save test evaluations
  - Make combined ROC
    - Specify whether or not to assume 100% accuracy on GC
    - Specify filename
  - Make Group-level ROC
    - Specify whether or not to assume 100% accuracy on GC
    - Specify filename