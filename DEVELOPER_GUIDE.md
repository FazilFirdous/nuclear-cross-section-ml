Developer Guide: Nuclear Cross-Section ML
=========================================

How every file works, why each design decision was made, and what to add next.

Written to be readable by someone learning both Python and nuclear physics.
Read this alongside the source code.


-------------------------------------------------------------------------------
WHAT THE PROJECT ACTUALLY DOES IN ONE SENTENCE
-------------------------------------------------------------------------------

We download experimental nuclear reaction data, compute a physics-informed
feature vector for each isotope (like its binding energy, spin, and shell-closure
distance), train gradient-boosted trees + a neural network to map features to
cross-sections, and then predict cross-sections for isotopes that have never been
measured.


-------------------------------------------------------------------------------
THE DATA FLOW (follow this top to bottom)
-------------------------------------------------------------------------------

1. scripts/download_data.py
       calls EXFORDownloader to hit IAEA REST API
       writes data/raw/exfor/n-gamma_all.parquet

2. scripts/build_dataset.py
       calls NuclearPropertiesLoader to parse AME2020 + NuBase2020
       calls DatasetBuilder to merge cross-sections with nuclear properties
       writes data/processed/n-gamma_dataset.h5

3. scripts/train_models.py
       reads the HDF5 file
       trains XGBoostCrossSectionModel, RandomForestCrossSectionModel, NeuralNetworkCrossSectionModel
       writes results/models/*.pkl
       writes results/figures/*.pdf

4. scripts/predict.py
       loads a .pkl model
       builds feature vectors for user-specified (Z, A, energy) combinations
       writes results/predictions/*.csv


-------------------------------------------------------------------------------
FILE-BY-FILE CODE WALKTHROUGH
-------------------------------------------------------------------------------

src/data/exfor_downloader.py
-----------------------------

CLASS: EXFORDownloader

  def __init__(self, output_dir, max_retries, retry_delay):
    Creates a requests.Session with a User-Agent header so the IAEA server
    knows who is calling. The session reuses TCP connections automatically.

  def _get_with_retry(self, url, params):
    Wraps every HTTP GET in a retry loop with exponential backoff:
      attempt 1 fails -> sleep 2s
      attempt 2 fails -> sleep 4s
      attempt 3 fails -> sleep 8s
    This is the standard pattern for any network code. Without it, a single
    dropped packet kills the entire download run.

  def search_exfor(self, reaction, quantity, z_min, z_max):
    Calls the IAEA NDS search endpoint. The returned JSON has a list of
    "entries" - each entry is one experiment, published in a journal, that
    measured a cross-section for some isotope.

  def download_entry(self, entry_id):
    Downloads one experiment's data. Caches it to disk as JSON so re-running
    the script doesn't re-download thousands of files.
    KEY DESIGN: cache-first pattern. Always check disk before network.

  def parse_entry_to_dataframe(self, entry_data):
    Flattens the nested JSON (experiment -> datasets -> data points) into
    a flat DataFrame where each row is one (energy, cross-section) measurement.
    The field names EN, DATA, ERR-T are the EXFOR standard names.

  def download_reaction(self, reaction, z_range, energy_range_eV):
    Orchestrates the full download: search -> iterate entries -> parse -> concat.
    Saves to Parquet (columnar format, 10x faster than CSV for numeric data).

  DICT: REACTION_MAP
    Maps human-readable names ("n-gamma") to EXFOR notation ("N,G").
    EXFOR uses CSEWG notation where comma separates projectile and ejectile.

FURTHER MODIFICATIONS YOU COULD ADD:
  - Add download of angular distribution data (MF=4 in ENDF)
  - Add download of resonance parameters (MF=2) for compound nucleus modeling
  - Add a --resume flag that skips already-downloaded entries
  - Connect to the IAEA EXFOR web API v2 (released 2024) for richer metadata


src/data/endf_processor.py
---------------------------

CLASS: ENDFProcessor

The ENDF-6 format is 50 years old and was designed for punch cards. It stores
numbers in fixed-width 11-character fields, 6 fields per line, 80 characters
total. The MF (file) number identifies data type; MF=3 is cross-sections.
The MT (reaction) number identifies the reaction type.

  def parse_endf_file(self, filepath):
    Reads the file, extracts Z and A from the header CONT record, finds all
    MF=3 blocks, parses each block as a TAB1 (tabulated 1D function) record.

  def _extract_mf3_blocks(self, lines):
    Scans line by line looking at columns 70-75 for MF and MT numbers.
    Groups all lines belonging to the same (MF=3, MT=x) block.

  def _parse_tab1_record(self, block_lines):
    ENDF TAB1 record structure:
      Line 1: CONT record (Z, A, meta-stable flag, etc.)
      Line 2: NR (number of interpolation ranges), NP (number of data pairs)
      Next NR lines: interpolation law for each range
      Next NP lines: (energy, cross-section) pairs
    We extract just the (E, sigma) pairs, skipping interpolation metadata.
    The replacement hack ('+' -> 'e+') handles ENDF's non-standard exponential
    notation: "1.23+05" means 1.23e+05.

  def compare_with_endf(self, ...):
    Log-log interpolation (np.interp on log10 axes) is used because cross-
    sections vary smoothly on log-log scales. Linear interpolation on linear
    axes would be wrong for this data.

FURTHER MODIFICATIONS:
  - Add MF=4 (angular distributions) and MF=5 (energy distributions) parsing
  - Add covariance data from MF=31-40 (experimental uncertainty matrices)
  - Support reading ENDF files directly from the NNDC tape download format
  - Add support for JEFF-3.3 and JENDL-5 libraries (same ENDF-6 format)


src/data/nuclear_properties.py
--------------------------------

CLASS: NuclearPropertiesLoader

  CONSTANTS: MAGIC_PROTON, MAGIC_NEUTRON
    The nuclear magic numbers are Z/N values where nuclear shells close,
    analogous to noble gas electron configurations. Nuclei with these
    proton or neutron counts are especially stable (enhanced binding energy,
    reduced cross-sections near thresholds). Magic numbers: 2, 8, 20, 28,
    50, 82, and for neutrons 126. Including proximity to magic numbers as
    features lets the model learn shell effects.

  def load_ame2020(self):
    Parses the AME2020 fixed-width text table. Each line represents one
    isotope; columns have strict fixed widths (documented in the AME paper).
    The "*" character in AME marks extrapolated (predicted) values.

  def _parse_spin_parity(self, sp_str):
    NuBase gives spin-parity strings like "3/2+", "0+", "(5/2-)" where:
      - "3/2" means spin J = 3/2 in units of hbar
      - "+" or "-" is parity under spatial reflection
      - Parentheses mean "uncertain"
    We convert spin to float (1.5 for "3/2") and parity to +1/-1.
    Half-integer spins come from odd numbers of nucleons.

  def _compute_derived_features(self, df):
    Most important derived features and WHY they predict cross-sections:

    isospin_asym = (N-Z)/A:
      Measures how far a nucleus is from N=Z (proton-neutron symmetric).
      Affects level density and hence compound nucleus cross-sections.

    delta_n (neutron pairing gap):
      Even-even nuclei have larger cross-sections for neutron capture because
      adding a neutron completes a pair (extra binding energy = bigger Q-value).
      Estimated as |S2n - 2*Sn|/2.

    dist_to_magic_Z/N:
      Cross-sections drop sharply at magic numbers (closed shells have fewer
      available final states). This feature encodes that shell effect.

    surface_area = A^(2/3):
      Nuclear radius scales as A^(1/3), so surface area as A^(2/3). The
      geometric cross-section sigma_geom = pi*R^2 provides a natural scale.

  def _estimate_features(self, Z, A):
    For isotopes not in AME (exotic nuclei), uses the Bethe-Weizsacker
    formula (liquid drop model). It has five terms:
      Volume:  +a_v * A          (bulk nuclear matter binding)
      Surface: -a_s * A^(2/3)   (surface nucleons are less bound)
      Coulomb: -a_c * Z(Z-1)/A^(1/3)  (proton repulsion)
      Asymmetry: -a_a*(N-Z)^2/A  (energy cost of N!=Z)
      Pairing:   +-a_p/sqrt(A)   (even-even bonus)

FURTHER MODIFICATIONS:
  - Add charge radii from the nuclear charge radii database (Angeli & Marinova)
  - Add Gamow-Teller strength from beta-decay databases (relevant for weak reactions)
  - Add level density parameters from RIPL-3 database
  - Use machine-learned nuclear masses (e.g., from Lovell 2020) for exotic nuclei
  - Add shell-model single-particle energies for nearby magic nuclei


src/data/dataset_builder.py
-----------------------------

CLASS: DatasetBuilder

  FEATURE_COLUMNS (the 22-element input vector):
    Z, N, A              - position on the nuclear chart (integer)
    Z_even, N_even       - odd/even parity (0 or 1); even-even nuclei have
                           different cross-section behavior from odd-A nuclei
    isospin_asym         - (N-Z)/A, see above
    surface_area         - A^(2/3), geometric scale
    binding_energy_per_A - B/A in MeV, peak ~8.8 MeV at Fe-56
    Sn, S2n, Sp, S2p    - one- and two-nucleon separation energies (MeV)
    delta_n, delta_p     - pairing gaps (MeV)
    spin, parity         - ground state quantum numbers
    beta2, beta4         - quadrupole and hexadecapole deformation
    dist_to_magic_Z/N   - shell closure proximity
    Q_value_MeV          - reaction energy release (determines threshold)
    log10_energy_eV      - log10 of incident neutron energy

    WHY log-transform the energy?
      Cross-sections vary over 10+ decades in energy. Without the log,
      a network layer trying to learn "this happens at 1 eV vs 1 MeV"
      sees numbers 1 and 1,000,000. After log10: 0 and 6. Much easier.

  TARGET_COLUMN = "log10_cross_section_barn":
    Same reason: cross-sections span 1e-8 to 1e6 barn. Log-space makes
    the regression problem well-behaved. Predicting in log space also
    enforces positivity (10^anything > 0).

  def _compute_q_values(self, df, reaction):
    Q-value for (n,gamma): Q = ME_target + ME_neutron - ME_compound
    where ME is the mass excess in keV (the deviation from A*u, u=atomic
    mass unit). Positive Q means energy is released (exothermic).
    For (n,gamma), Q is always positive (adding a neutron releases energy).

  def _bin_energy_grid(self, df, bins_per_decade):
    Problem: some isotopes like Au-197 have 1000+ measurements at closely
    spaced energies; others have 3. Without binning, Au-197 would dominate
    the training loss just due to measurement density, not physics.
    Solution: divide the energy axis into log-spaced bins; in each bin keep
    only the measurement with the smallest uncertainty.

  def split_train_val_test(self, df, strategy="leave_one_isotope_out"):
    CRITICAL DESIGN CHOICE: leave_one_isotope_out strategy.
    Standard random split would put neighboring energy points of the same
    isotope in both train and test. The model would just interpolate and
    score perfectly on test while failing on truly unmeasured isotopes.
    LOIO forces the model to generalize ACROSS isotopes, which is the
    actual scientific goal.

FURTHER MODIFICATIONS:
  - Add leave-one-mass-region-out splitting for heavier-tailed evaluation
  - Add data augmentation: perturb features within experimental uncertainties
  - Add sample weighting by experimental uncertainty (downweight poor data)
  - Add feature for "distance to nearest measured isotope" on the nuclear chart


src/models/xgboost_model.py
-----------------------------

CLASS: XGBoostCrossSectionModel

XGBoost = Extreme Gradient Boosting. An ensemble of decision trees where
each tree corrects the residuals of the previous tree.

  UNCERTAINTY: quantile regression
    Instead of just predicting the mean, we train 5 separate models:
      alpha=0.05 -> predicts the 5th percentile (lower 90% bound)
      alpha=0.16 -> predicts the 16th percentile (lower 68% bound)
      alpha=0.50 -> predicts the median
      alpha=0.84 -> predicts the 84th percentile (upper 68% bound)
      alpha=0.95 -> predicts the 95th percentile (upper 90% bound)
    The 68% interval [q16, q84] corresponds to a ±1 sigma Gaussian band.
    Quantile regression minimizes the "pinball loss" rather than squared error:
      loss = alpha*(y-q)  if y > q  (penalize underestimating)
      loss = (1-alpha)*(q-y) if y <= q  (penalize overestimating)

  KEY HYPERPARAMETERS:
    n_estimators=1000:    Number of trees. More = better fit but slower.
    max_depth=6:          Maximum depth per tree. Controls overfitting.
    learning_rate=0.05:   Small steps + many trees = better generalization.
    subsample=0.8:        Each tree sees 80% of training rows (stochastic).
    colsample_bytree=0.8: Each tree sees 80% of features (like Random Forest).
    min_child_weight=5:   Minimum samples required to split a node. Prevents
                          fitting to noise in small groups.
    early_stopping_rounds=50: Stop if validation error doesn't improve for
                              50 consecutive trees.

  def feature_importance(self):
    "Gain" importance: total information gain from all splits using this feature.
    "Cover" importance: average number of samples affected by splits on this feature.
    Use gain for "which feature is most predictive?" and cover for "which feature
    affects the most data points?".

FURTHER MODIFICATIONS:
  - Add SHAP values for instance-level explainability (pip install shap)
  - Add LightGBM as an alternative (usually faster, similar accuracy)
  - Add CatBoost which handles categorical features (element names) natively
  - Use Bayesian hyperparameter optimization (Optuna) instead of grid search
  - Add conformal prediction for distribution-free coverage guarantees


src/models/random_forest_model.py
-----------------------------------

CLASS: RandomForestCrossSectionModel

Random Forest = ensemble of independent decision trees, each trained on a
bootstrap sample (random sample with replacement) of the training data.

  UNCERTAINTY: ensemble variance
    The standard deviation across individual tree predictions is a natural
    uncertainty estimate. Trees trained on different bootstrap samples will
    disagree more for regions of feature space with few training examples
    (exactly the exotic isotopes we want to predict).

  predict_with_uncertainty():
    Collects predictions from all 500 trees individually (self._model.estimators_).
    This is memory-intensive but gives the full predictive distribution.
    For large test sets, consider batching.

  oob_score:
    Out-of-bag score: each tree's out-of-bag samples (the ~37% of training
    data not in its bootstrap sample) form a free internal validation set.
    This is a reliable estimate of test performance without using held-out data.

FURTHER MODIFICATIONS:
  - Add quantile regression forests (QuantileForestRegressor in scikit-learn)
    for better-calibrated intervals than variance-based estimates
  - Add partial dependence plots (sklearn.inspection.PartialDependenceDisplay)
    to visualize how each feature affects predictions
  - Add a proximity matrix to identify training points most similar to test isotopes
  - Implement the Jackknife-after-bootstrap variance estimator (Wager et al. 2014)
    for more theoretically grounded uncertainty


src/models/neural_network.py
------------------------------

CLASS: CrossSectionNet (the PyTorch nn.Module)

Architecture: Input -> [Linear -> BN -> ReLU -> Dropout -> ResBlock] x N -> Output

  ResidualBlock:
    The key idea: output = ReLU(x + f(x)) where f is the transformation.
    The skip connection (identity shortcut) solves the vanishing gradient problem
    in deep networks: gradients flow directly back to early layers through the
    addition. Without it, gradients shrink exponentially with network depth.

  BatchNorm1d:
    Normalizes activations within each mini-batch: subtract mean, divide by std.
    Benefits: (1) faster training (larger learning rates stable), (2) acts as
    regularization (because batch statistics add noise), (3) reduces sensitivity
    to weight initialization.

  Dropout:
    Randomly zeros a fraction `p` of neurons during training. This is Bernoulli
    noise injection. It prevents co-adaptation: no neuron can rely on any specific
    other neuron being active, so each learns more robust features.

CLASS: NeuralNetworkCrossSectionModel (the training wrapper)

  MC DROPOUT UNCERTAINTY (Gal & Ghahramani 2016):
    At test time, we call model.enable_dropout() to keep dropout ACTIVE (normally
    it is disabled in eval mode). Then we run 100 forward passes on the same input.
    Each pass gives a different prediction because different neurons are dropped.
    The MEAN of 100 predictions is our point estimate.
    The STD of 100 predictions is our uncertainty estimate (epistemic uncertainty).
    This approximates sampling from a Bayesian neural network posterior.

  Training loop:
    for each epoch:
      for each mini-batch:
        zero_grad()              <- clear accumulated gradients
        y_pred = model(X_batch)  <- forward pass
        loss = MSELoss(y_pred, y_batch)  <- compute loss
        loss.backward()          <- compute gradients (backprop)
        clip_grad_norm_()        <- cap gradient magnitude (prevents explosions)
        optimizer.step()         <- update weights: w -= lr * grad

  ReduceLROnPlateau scheduler:
    If validation loss doesn't improve for 10 epochs, halve the learning rate.
    This allows large initial learning rates (fast early training) then fine-
    tuning with small steps.

  Early stopping:
    Tracks best validation loss and restores best weights at the end.
    Without this, the model would overfit the training data in later epochs.

FURTHER MODIFICATIONS:
  - Replace MC dropout with Deep Ensembles (train 5 independent networks;
    take mean and variance across them) for better-calibrated uncertainty
  - Add attention mechanism over the feature vector to weight features dynamically
  - Implement a Graph Neural Network where isotopes are nodes and nuclear
    structure similarity defines edges (message-passing between neighboring nuclei)
  - Add physics-informed loss: penalize predictions that violate the compound
    nucleus unitarity constraint (sum of partial cross-sections <= total)
  - Add a normalizing flow as the output for full predictive density (not just mean+std)


src/models/ensemble.py
-----------------------

CLASS: EnsembleCrossSectionModel

Stacking ensemble: a "meta-learner" (Ridge regression) combines the three
base model predictions. The meta-learner learns the optimal weights by
minimizing error on the validation set (which none of the base models saw
during training).

  fit_meta_learner(X_val, y_val):
    Runs all three models on the validation set, creates a matrix of shape
    (n_val, 3) where column j is model j's predictions, then fits:
      y_val ≈ a0*xgb_pred + a1*rf_pred + a2*nn_pred + bias
    The Ridge penalty (L2 regularization) prevents any single model from
    getting a very large negative weight just because it has low correlation
    with another model.

  TOTAL UNCERTAINTY DECOMPOSITION:
    std_epistemic: disagreement between models = uncertainty from limited data.
      When all three models agree, we are confident. When they disagree widely
      (e.g., for exotic isotopes far from training data), uncertainty is high.
    std_aleatoric: internal uncertainty from each model (MC dropout, quantile width).
      This is irreducible noise in the data itself (measurement uncertainty).
    std_total = sqrt(epistemic^2 + aleatoric^2): quadrature sum.

FURTHER MODIFICATIONS:
  - Replace Ridge meta-learner with a gradient-boosted meta-learner
  - Add model selection: automatically discard models with >2x median RMSE
  - Implement temperature scaling on the combined uncertainty for better calibration
  - Add a Bayesian model averaging scheme (weight by marginal likelihood)


src/evaluation/cross_validation.py
-------------------------------------

CLASS: LeaveOneIsotopeOutCV

This is the scientifically correct validation strategy for this problem.

  WHY NOT STANDARD K-FOLD:
    In standard 5-fold CV, if Au-197 measurements at 1 eV and 2 eV are in
    different folds, the model predicts "Au-197 at 2 eV" having seen Au-197
    at many other energies. That is interpolation, not extrapolation.
    We want to test: "can the model predict Au-197 having never seen Au-197?"

  split(df):
    Creates a generator (lazy evaluation with yield). Each call to next()
    returns the next (train_indices, test_indices) pair without storing all
    splits in memory simultaneously.

  run_loio_cv(model_class, ...):
    Importantly, the FeatureEngineer (imputer + scaler) is RE-FIT on each
    training fold. This is critical: if we fit the scaler on the full dataset
    (including test), we leak information about the test set into the model.
    Each fold's scaler/imputer must be fit only on that fold's training data.

FURTHER MODIFICATIONS:
  - Add leave-one-element-out (all isotopes of element Z held out)
    to test generalization across chemical elements
  - Add "extrapolation distance" metric: how far (in Z-N space) is the
    test isotope from the nearest training isotope?
  - Implement temporal splitting: train on pre-2000 measurements, test on post-2000


src/evaluation/metrics.py
---------------------------

  compute_metrics():
    Standard metrics:
      RMSE = sqrt(mean((y_pred - y_true)^2))  <- sensitive to outliers
      MAE  = mean(|y_pred - y_true|)          <- robust to outliers
      R^2  = 1 - SS_res/SS_tot               <- fraction of variance explained
      MBE  = mean(y_pred - y_true)            <- systematic bias

    Nuclear-specific metrics:
      fraction_within_factor2: fraction where |log10(pred/true)| < log10(2).
        In nuclear physics, "within a factor of 2" is the standard benchmark.
      fraction_within_factor10: looser criterion for exotic regions.

    Interval metrics:
      PICP (Prediction Interval Coverage Probability): fraction of true values
        inside the predicted interval. Should equal the nominal coverage
        (e.g., 68% intervals should contain 68% of true values).
      MPIW (Mean Prediction Interval Width): average width. Narrower is better
        given correct coverage.
      Winkler score: single number combining coverage and width. Lower is better.

  calibration_error():
    For a Gaussian uncertainty model with predicted std sigma_i:
      z_i = (y_i - mu_i) / sigma_i should follow a standard normal distribution.
    We check this by computing what fraction of z_i falls within +-z_alpha for
    a range of alpha values. If coverage matches expected, the model is calibrated.
    Expected Calibration Error (ECE) = mean absolute deviation from the diagonal.

FURTHER MODIFICATIONS:
  - Add the NIG (Normal-Inverse-Gamma) distribution metrics for deep evidential regression
  - Add isotope-level metrics (average by isotope, not by data point) to avoid
    well-measured isotopes dominating the scores
  - Add energy-dependent metrics: is the model more accurate at thermal or fast energies?


src/visualization/plots.py
----------------------------

All figures use:
  serif font: matches Physics journal style
  300 DPI: minimum for publication submission
  ColorBrewer palette: colorblind-safe (tested against 8% of males with color deficiency)
  No chart junk: no grid background, minimal tick marks (Tufte data-ink ratio principle)

  plot_predicted_vs_experimental():
    The "parity plot" is the standard nuclear data benchmark figure.
    Factor-of-2 shaded band: the conventional acceptability threshold.
    Log-log scale implicitly because both axes are log10-transformed.
    RMSE and R^2 in text box: standard metrics for the caption.

  plot_nuclear_chart_heatmap():
    N on x-axis, Z on y-axis: this is the conventional Segre chart orientation.
    Magic number dotted lines at 2, 8, 20, 28, 50, 82, 126.
    s=15 square markers: approximate the pixel size for isotope cells.

  plot_cross_section_energy_dependence():
    Energy in eV (not MeV) for consistency with EXFOR convention.
    Log-log axes: cross-sections are smooth on this scale.
    Three reference curves (ENDF, TALYS, ML) on same axes for comparison.

  plot_uncertainty_calibration():
    "Reliability diagram": standard visualization for probabilistic model calibration.
    Above diagonal = overconfident (intervals too narrow).
    Below diagonal = underconfident (intervals too wide).
    For nuclear data, it is safer to be slightly underconfident.

FURTHER MODIFICATIONS:
  - Add animated GIF/video of cross-section evolution along an isotopic chain
  - Add interactive Plotly charts for the notebooks (plotly.express.scatter with hover)
  - Add Segre chart with nuclide half-lives as background reference
  - Add comparison of multiple reactions on the same nuclear chart


-------------------------------------------------------------------------------
WHERE IS THE MAIN PART OF THE PROJECT?
-------------------------------------------------------------------------------

The most important files, in order of scientific significance:

1. src/data/dataset_builder.py (FEATURE_COLUMNS + build())
   This is where nuclear physics knowledge is encoded as ML features.
   The choice of features determines everything downstream.

2. src/models/xgboost_model.py (quantile regression for UQ)
   This will be your best-performing model in most regimes.
   XGBoost dominates structured-data ML benchmarks consistently.

3. src/evaluation/cross_validation.py (LeaveOneIsotopeOutCV)
   This is where the scientific validity of the results is established.
   A wrong validation strategy would produce misleadingly good numbers.

4. src/data/nuclear_properties.py (NuclearPropertiesLoader)
   The connection to the nuclear physics literature.
   Feature quality matters more than model architecture for this problem.


-------------------------------------------------------------------------------
WHAT TO BUILD NEXT (in recommended order)
-------------------------------------------------------------------------------

PHASE 1: Get real data working (1-2 weeks)
  - Run download_data.py to pull EXFOR (n,gamma) data
  - Download AME2020 and NuBase2020 from IAEA NDS manually if needed
  - Run build_dataset.py and inspect the feature matrix
  - Train XGBoost on the real data and examine feature importances

PHASE 2: Improve feature engineering (1-2 weeks)
  - Add nuclear charge radii as a feature (strongly correlated with cross-sections)
  - Add level density parameters from RIPL-3 for compound nucleus reactions
  - Add the compound nucleus mass (Z, A+1) properties as features
  - Experiment with polynomial and interaction features (Z*N, Z^2/A^(1/3))

PHASE 3: Better uncertainty quantification (2-3 weeks)
  - Implement Deep Ensembles (5 independent neural networks; better calibrated
    than MC dropout according to Lakshminarayanan et al. 2017)
  - Add conformal prediction (guaranteed coverage without distributional assumptions)
  - Add SHAP values for per-prediction feature attribution

PHASE 4: Physics-informed constraints (3-4 weeks)
  - Enforce optical model sum rules as soft constraints in the loss function
  - Add Hauser-Feshbach TALYS predictions as additional input features
  - Implement a multi-task model predicting all reaction channels simultaneously
    (n-total, n-elastic, n-gamma, n-fission share physical constraints)

PHASE 5: Astrophysics applications (4-6 weeks)
  - Generate predictions for the full r-process path (neutron-rich isotopes
    from Z=26 to Z=83, N=50 to N=126)
  - Compare with BRUSLIB and REACLIB stellar rate libraries
  - Compute stellar reaction rates by folding cross-sections with
    Maxwell-Boltzmann energy distribution: <sigma*v> = integral sigma(E) * v * f(E,T) dE

PHASE 6: Publication preparation
  - Run full LOIO-CV on all reaction types and mass regions
  - Generate all comparison figures against ENDF/B-VIII.0 and TALYS
  - Write Methods section based on this codebase
  - Submit to Physical Review C or European Physical Journal A


-------------------------------------------------------------------------------
KEY PYTHON PATTERNS USED IN THIS PROJECT
-------------------------------------------------------------------------------

1. DATACLASS PATTERN (each class is self-contained):
   Every model class has: __init__, fit, predict, predict_with_uncertainty,
   evaluate, save, load. You can swap any model without changing other code.

2. GENERATOR PATTERN (cross_validation.py):
   def split(self, df): ... yield train_idx, test_idx
   Using yield instead of return makes the CV loop memory-efficient because
   only one split exists in memory at a time.

3. CACHE-FIRST PATTERN (exfor_downloader.py):
   if cache_path.exists(): return load(cache_path)
   data = download(); save(data, cache_path); return data
   Always check disk before network. Saves hours on re-runs.

4. OPTIONAL DEPENDENCIES (ENDF parser):
   The project works without openmc or ENDFtk installed. Optional features
   degrade gracefully with warnings rather than crashing.

5. TYPE HINTS:
   def predict(self, X: np.ndarray) -> np.ndarray:
   These are not enforced at runtime but serve as documentation and enable
   IDE autocomplete and type checkers (mypy).

6. LOGURU OVER LOGGING:
   logger.info("thing: %d", count) instead of print()
   Gives timestamps, log levels, file/line numbers automatically.
   In scripts: logger.remove(); logger.add(sys.stderr, level="INFO")
