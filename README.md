Resource-Aware Automata Selection System (RAAS)

🌿 Green Computing meets Theoretical Computer Science
This framework optimizes the execution of formal languages (Regular, Context-Free, and Context-Sensitive) based on physical hardware energy consumption. By benchmarking **Deterministic Finite Automata (DFA)** and **Pushdown Automata (PDA)** on actual hardware, the system trains an **XGBoost Classifier** to predict the most energy-efficient computational model for any given input string.


🚀 System Architecture
The system operates across four distinct phases:

1.  **Dataset Generation**: Creates a balanced dataset spanning the Chomsky Hierarchy (Parity, Dyck, and $a^n b^n c^n$ patterns).
2.  **Hardware Benchmarking**: Utilizes **Intel RAPL (Running Average Power Limit)** via `pyrapl` to measure real-time CPU energy (Joules) on HP Omen hardware.
3.  **Noise Injection**: Introduces "Strategic Noise"—cases where theoretical models are outperformed by simpler machines due to physical overheads like the **"Stack Initialization Tax"**.
4.  **Predictive Modeling**: An XGBoost model learns the relationship between 14 structural features and physical energy draw to select the optimal machine.


📊 Dataset Schema
The model utilizes a 14-column dataset to make selection decisions:

Column Name	             Type	          Description
sequence	               String	        The raw input string (e.g., 0101 or ((()))).
label	                   Int	          Theoretical class (0: Regular, 1: CFG, 2: CSG).
language_name            String	        Sub-type (e.g., parity, dyck, an_bn_cn).
alphabet_size	           Int	          Count of unique symbols in the string.
rule_count            	 Int	          Grammar rules required for the language.
max_nesting_depth        Int	          Deepest stack level reached during processing.
avg_string_length        Float	        Mean length of strings in the specific language batch.
is_ambiguous          	 Bool	          Whether the string allows multiple parse trees.
complexity	             Int	          Estimated computational difficulty.
dfa_energy	             Float	        Measured energy for DFA execution (Joules).
pda_energy	             Float	        Measured energy for PDA execution (Joules).
dfa_state	               Int	          Total states visited in the DFA.
pda_stack	               Int	          Total stack operations performed in the PDA.
optimal_model	           Target	        The Greenest choice: 0 (DFA), 1 (PDA), 2 (TM).


🛠 Installation & Usage

1. Environment Setup

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas xgboost scikit-learn matplotlib seaborn pyrapl pyformlang


2. Execution Pipeline
# Step 1: Benchmark Hardware (Requires sudo for Intel RAPL access)
sudo ./venv/bin/python benchmarker.py

# Step 2: Inject Noise (To simulate real-world energy overlap)
python inject_noise.py

# Step 3: Train the XGBoost Selector
python train_model.py

# Step 4: Run Real-time Inference
python predict.py


Experimental Results

Model Performance
After training on 30,000+ hardware-benchmarked samples and injecting strategic noise, the XGBoost selector achieved the following metrics:

Overall Accuracy: 97.98%

Precision (DFA): 0.98

Recall (PDA): 0.98

F1-Score: 0.98

The 2.02% error margin represents the Decision Uncertainty Zone, where hardware fluctuations and CPU thermal throttling make the energy profiles of DFAs and PDAs nearly identical.

Feature Importance (The Decision Brain)
The XGBoost model identified the following features as the primary drivers for energy-aware selection:

max_nesting_depth: The strongest predictor of PDA energy overhead.

complexity: Determines the baseline computational path.

is_ambiguous: Key indicator for potential branching energy spikes in non-deterministic scenarios.

alphabet_size: Correlates with state-transition table memory width.

Energy Savings Impact
By using this predictive selector instead of defaulting to a Universal Turing Machine or a standard PDA, the system reduces average power consumption by:

~12-15% for Regular Language strings processed via DFA instead of PDA.

~30% for short Context-Free strings where the Stack Tax was avoided.
