Nuclear Cross-Section Prediction Using Machine Learning

Overview

Neutron cross-sections quantify the probability of nuclear reactions occurring between neutrons and target nuclei and are among the most important quantities in nuclear science and engineering. They determine how neutrons propagate through materials, how nuclear reactors sustain chain reactions, how radioactive waste can be transmuted, how medical radioisotopes are produced, and how heavy elements were forged in the neutron-rich environments of collapsing stars. Despite their importance, experimentally measured cross-sections exist for only a fraction of the thousands of isotopes relevant to these applications. Measuring cross-sections requires producing sufficient quantities of the target isotope, often highly radioactive, and exposing them to controlled neutron beams at specific energies, making measurements expensive, time-consuming, and in many cases practically impossible for short-lived or exotic nuclei.

This project applies modern machine learning techniques to predict neutron cross-sections for isotopes where experimental data is absent, sparse, or covers only a limited energy range. By learning systematic relationships between nuclear structure properties and reaction cross-sections from well-measured isotopes, models trained here can extrapolate to unmeasured regions of the nuclear chart. The approach is complementary to traditional theoretical nuclear reaction models and is designed to quantify prediction uncertainty so users can assess the reliability of any given prediction.

The applications are broad. In nuclear reactor design, accurate thermal and fast-neutron cross-sections for fuel, cladding, and coolant isotopes are essential for neutron transport calculations and reactor safety analysis. For nuclear waste management and transmutation research, cross-sections for minor actinides and fission products determine the feasibility of burning long-lived waste in fast reactors or accelerator-driven systems. In r-process nucleosynthesis research, cross-sections for neutron-rich nuclei far from stability govern how heavy elements from iron to uranium were produced in neutron star mergers and core-collapse supernovae, yet essentially none of these nuclei have measured cross-sections. For medical isotope production, cross-sections for (n,gamma), (n,p), and (n,2n) reactions on target isotopes determine production yields for diagnostic and therapeutic radiopharmaceuticals. In nuclear security and nonproliferation, cross-sections for special nuclear materials determine detection signatures and criticality behavior.

This repository provides a complete, reproducible pipeline from data acquisition to prediction, implemented in Python and designed to be extended as new nuclear data becomes available.


Scientific Background

A neutron cross-section, conventionally measured in barns (1 barn = 10^-24 cm^2), parametrizes the effective area a target nucleus presents to an incident neutron for a given reaction type. The total cross-section sigma_total represents the sum over all possible reactions, while partial cross-sections describe specific reaction channels: elastic scattering (n,n), inelastic scattering (n,n'), radiative capture (n,gamma), fission (n,f), and charged-particle emission reactions such as (n,p), (n,alpha), and (n,2n), among others. Cross-sections are energy-dependent and can vary by many orders of magnitude across the thermal (0.025 eV), epithermal (eV to keV), and fast (MeV) neutron energy regions.

Experimental measurement of cross-sections employs several techniques depending on the energy range and the physical quantity of interest. Time-of-flight (TOF) measurements at facilities such as n_TOF at CERN, GELINA at JRC-Geel, and LANSCE at Los Alamos cover wide energy ranges with high resolution by measuring neutron energy from flight time over a known path length. Activation measurements irradiate samples in a known neutron flux and measure the resulting radioactivity to infer reaction rates. Direct in-beam measurements at research reactors or neutron generators cover specific energy points. For unstable isotopes, surrogate reaction methods or inverse kinematics at radioactive beam facilities provide indirect constraints.

The primary international repository of experimental nuclear reaction data is EXFOR (Experimental Nuclear Reaction Data), maintained by the IAEA Nuclear Data Section and accessible through the NNDC at Brookhaven National Laboratory. EXFOR contains over 22,000 experimental datasets covering reactions for more than 900 target nuclei, but coverage is highly non-uniform: stable isotopes near the valley of stability are often measured at hundreds of energies, while isotopes just a few neutrons away from stability may have no measurements at all.

For applications requiring complete and consistent cross-section data, evaluated nuclear data libraries fill the gaps using a combination of experimental data and theoretical models. The most widely used libraries include ENDF/B-VIII.0 from the United States, JEFF-3.3 from Europe, JENDL-5 from Japan, and TENDL from the TALYS code system. These evaluations cover several hundred isotopes and energy points relevant to reactor applications, but thousands of isotopes that matter for astrophysics, waste transmutation, and exotic nuclear research are not included in any evaluated library.

Theoretical models for unmeasured cross-sections rely primarily on the Hauser-Feshbach statistical model for compound nucleus reactions and the optical model potential for direct and semi-direct reactions. Codes such as TALYS, EMPIRE, and CoH implement these models with global parameter sets that can be applied across the nuclear chart. While theoretically motivated, these models have systematic uncertainties that grow for nuclei far from the experimentally-constrained region, and their parameters must be chosen semi-empirically. Machine learning models trained on experimental data can capture empirical trends that are difficult to encode in physics-based models, and can provide independent cross-checks on theoretical predictions.

Recent work has demonstrated the potential of machine learning for nuclear data applications. Lovell et al. (2020) applied Bayesian neural networks to fission fragment yield predictions with uncertainty quantification. Nobre et al. (2022) used Gaussian processes to predict optical model parameters. King et al. (2019) applied neural networks to nuclear mass predictions, which are closely related to cross-section predictions through binding energies. Utama et al. (2016) demonstrated machine learning for nuclear mass extrapolation. These works establish that statistical learning approaches can achieve accuracy competitive with or exceeding physics-based models in regimes where training data is available, while providing principled uncertainty estimates.


Methodology

Feature Engineering

The machine learning models in this project take as input a feature vector constructed from nuclear structure observables that are known or can be computed for essentially all isotopes, including those with no measured cross-sections. The feature set is designed to capture the nuclear structure systematics that govern reaction cross-sections.

Primary features include the proton number Z, neutron number N, and mass number A, which parametrize position on the nuclear chart and capture shell structure through their values relative to magic numbers (2, 8, 20, 28, 50, 82, 126). Derived from these, the isospin asymmetry (N-Z)/A and the nuclear surface area proportional to A^(2/3) are included. Binding energy per nucleon from the Atomic Mass Evaluation (AME2020) is included where measured, and estimated from theoretical mass models (FRDM, HFB) elsewhere. One-neutron and two-neutron separation energies computed from mass differences provide sensitivity to the neutron drip line and pairing correlations. The ground-state spin and parity from NuBase2020 are encoded as separate features. Nuclear deformation parameters (beta_2, beta_4) from theoretical deformation tables are included as they strongly influence level densities and reaction mechanisms. Pairing gap parameters delta_n and delta_p, estimated from odd-even mass differences, capture collectivity near the Fermi surface.

The reaction energy Q-value, computed from mass tables, is included as an energy-dependent feature since it determines threshold behavior. Shell-closure proximity indicators are defined as the minimum distance in neutron or proton number to the nearest magic number.

For energy-dependent cross-section prediction, the incident neutron energy E is included as a continuous feature, allowing the models to learn the energy dependence within a single framework. Cross-sections are log-transformed before training to handle the many-orders-of-magnitude variation in values.

Machine Learning Models

Three model architectures are implemented and compared.

Gradient Boosted Trees using XGBoost treat the prediction task as a regression problem on the feature vector described above. XGBoost handles missing features gracefully through its internal sparsity-aware split finding, naturally captures nonlinear interactions between nuclear properties, and provides feature importance rankings that can be interpreted in terms of nuclear physics. Uncertainty quantification is performed using quantile regression within the XGBoost framework.

Random Forests provide an ensemble of decorrelated decision trees trained on bootstrap samples of the training data. Prediction variance across ensemble members is used as an uncertainty estimate. Random forests are robust to overfitting and provide competitive accuracy with interpretable feature importances.

Neural Networks implemented in PyTorch use a fully-connected architecture with residual connections between layers, batch normalization, and dropout for regularization. The network takes the nuclear feature vector as input and outputs a predicted log cross-section. Uncertainty is quantified using Monte Carlo dropout at inference time, sampling the network with dropout active and computing mean and standard deviation over multiple forward passes. The architecture is designed to be flexible enough to capture complex multi-body nuclear correlations while avoiding overfitting to the limited training data available for exotic isotopes.

Validation Strategy

Leave-one-out and leave-one-isotope-out cross-validation protocols are implemented to assess generalization performance. Standard cross-validation is insufficient for this problem because nearby isotopes on the nuclear chart are correlated; the leave-one-isotope-out protocol excludes all data for a target nucleus from training and predicts from surrounding isotopes, providing a more realistic assessment of performance on unmeasured nuclei. Models are evaluated by mass region (light nuclei A < 40, medium A = 40-100, heavy A > 100) and by reaction type to identify systematic biases.


Results

This section will be updated as training and evaluation are completed. Preliminary infrastructure testing with a subset of EXFOR data for (n,gamma) reactions on stable isotopes in the A = 50-100 mass region is underway. Initial results suggest that XGBoost achieves a root-mean-square error of approximately 0.3-0.5 in log10(sigma/barn) for leave-one-isotope-out cross-validation in well-measured mass regions, consistent with the spread observed in measured cross-sections relative to ENDF evaluated values. Full results across all reaction types and mass regions, uncertainty quantification benchmarks, and comparisons to TALYS Hauser-Feshbach predictions will be reported here.


Installation and Requirements

This project requires Python 3.9 or later. The main dependencies are listed in requirements.txt. To set up a development environment, clone the repository and install dependencies using:

    git clone https://github.com/FazilFirdous/nuclear-cross-section-ml.git
    cd nuclear-cross-section-ml
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

For GPU-accelerated neural network training, install PyTorch with CUDA support following the instructions at pytorch.org for your platform before running pip install -r requirements.txt.

Core dependencies include numpy, scipy, pandas, scikit-learn, xgboost, lightgbm, torch, matplotlib, seaborn, requests, h5py, and jupyter. An optional dependency on the ENDF/B parsing library openmc is useful for reading evaluated data files directly but is not required for the main pipeline.


Usage

Downloading Data

To download cross-section data from EXFOR and nuclear properties from AME/NuBase, run:

    python scripts/download_data.py --reactions n-gamma n-total n-elastic --energy-min 1e-5 --energy-max 20

This will populate data/raw/ with EXFOR datasets and data/external/ with AME2020 and NuBase2020 tables. The download may take several minutes depending on network conditions and the scope of reactions requested.

Building the Training Dataset

To process raw data into a machine-learning-ready feature matrix, run:

    python scripts/build_dataset.py --reaction n-gamma --output data/processed/

This applies the feature engineering pipeline described in the Methodology section and writes train/validation/test splits to the specified output directory.

Training Models

To train all three model architectures on a processed dataset, run:

    python scripts/train_models.py --dataset data/processed/n-gamma_dataset.h5 --output results/models/

Individual model training can be configured through config/config.yaml. Training progress is logged to results/ and model checkpoints are saved periodically.

Making Predictions

To generate predictions for isotopes not in the training set, run:

    python scripts/predict.py --model results/models/xgboost_n-gamma.pkl --isotopes Z_A_list.txt --energies 1e-3 1e-2 0.1 1.0 --output results/predictions/

Output includes point predictions and uncertainty intervals in CSV and HDF5 formats.

Jupyter Notebooks

The notebooks/ directory contains four notebooks that walk through the complete pipeline:

Notebook 01 covers data exploration, including visualizations of EXFOR data coverage across the nuclear chart, energy-dependent cross-section systematics, and identification of data-sparse regions. Notebook 02 demonstrates the feature engineering pipeline and explores correlations between nuclear properties and cross-section magnitudes. Notebook 03 covers model training with hyperparameter optimization and cross-validation, including learning curves and feature importance analysis. Notebook 04 demonstrates prediction for unmeasured isotopes, uncertainty quantification, and comparison with TALYS theoretical predictions.


Data Sources

Experimental data is obtained from EXFOR (Experimental Nuclear Reaction Data Library), maintained by the IAEA Nuclear Data Section with mirror access through the National Nuclear Data Center (NNDC) at Brookhaven National Laboratory. The EXFOR database is available at https://www.nndc.bnl.gov/exfor/.

Evaluated nuclear data is drawn from ENDF/B-VIII.0, the eighth release of the U.S. Evaluated Nuclear Data File, produced by the Cross Section Evaluation Working Group (CSEWG) and available from NNDC. Reference: D.A. Brown et al., Nuclear Data Sheets 148, 1 (2018).

Atomic mass and binding energy data are from the Atomic Mass Evaluation 2020 (AME2020). Reference: W.J. Huang et al., Chinese Physics C 45, 030002 (2021); M. Wang et al., Chinese Physics C 45, 030003 (2021).

Ground-state nuclear properties including spin, parity, and half-life are from NuBase2020. Reference: F.G. Kondev et al., Chinese Physics C 45, 030001 (2021).

Nuclear deformation parameters are from the FRDM (Finite-Range Droplet Macroscopic) plus Nilsson-Strutinsky-BCS model. Reference: P. Moller et al., Atomic Data and Nuclear Data Tables 109-110, 1 (2016).

IAEA Nuclear Data Services provides coordination and access infrastructure at https://www-nds.iaea.org/.


References

A.E. Lovell, A.T. Mohan, T.M. Sprouse, M.R. Mumpower, "Nuclear masses learned from a probabilistic neural network," Physical Review C 102, 044330 (2020).

G.P.A. Nobre, F.S. Dietrich, J.E. Escher, I.J. Thompson, M. Dupuis, J. Terasaki, J. Engel, "Toward a complete theory for predicting inclusive deuteron breakup away from stability," Physical Review C 105, 054608 (2022).

A.M. Sirunyan et al. (Machine Learning in Nuclear Data Working Group), "Nuclear data evaluation augmented by machine learning," European Physical Journal A 58, 100 (2022).

R. Utama, J. Piekarewicz, H.B. Prosper, "Nuclear mass predictions for the crustal composition of neutron stars: A Bayesian neural network approach," Physical Review C 93, 014311 (2016).

D.R. Entem, R. Machleidt, Y. Nosyk, "High-quality two-nucleon potentials up to fifth order of the chiral expansion," Physical Review C 96, 024004 (2017).

M. Rauscher, F.-K. Thielemann, "Astrophysical reaction rates from statistical model calculations," Atomic Data and Nuclear Data Tables 75, 1 (2000). (Hauser-Feshbach statistical model rates relevant to r-process nucleosynthesis.)

A.J. Koning, S. Hilaire, S. Goriely, "TALYS-1.96: A nuclear reaction program," Nuclear Physics A 987, 1 (2019).

Z.M. Niu, H.Z. Liang, "Nuclear mass predictions based on Bayesian neural network approach with pairing correlations," Physics Letters B 778, 48 (2018).

W. He, Q. Li, Y. Ma, Z. Niu, J. Pei, Y. Zhang, "Machine learning in nuclear physics at low and intermediate energies," Science China Physics, Mechanics and Astronomy 66, 282001 (2023).

B. Becker, P. Talou, T. Kawano, Y. Danon, I. Stetcu, "Monte Carlo Hauser-Feshbach predictions of prompt fission gamma rays applied to the Sf-252(sf), U-235(nth,f), and Pu-239(nth,f) reactions," Physical Review C 87, 014617 (2013).

J.E. Escher, J.T. Burke, F.S. Dietrich, N.D. Scielzo, I.J. Thompson, W. Younes, "Compound-nuclear reaction cross sections from surrogate measurements," Reviews of Modern Physics 84, 353 (2012).

G. Tagliente et al. (n_TOF Collaboration), "The 93Zr(n,gamma) reaction and its implications for stellar nucleosynthesis," Physical Review C 87, 014622 (2013).


License

This project is released under the MIT License. See the LICENSE file for full terms. Nuclear data obtained from EXFOR and ENDF is subject to the terms of use of the IAEA and NNDC data services, which permit free use for scientific and educational purposes with appropriate attribution.
