# PATHSNAP-ML-Hackathon
## Member 1: data preparation and road alignment

See [member1/README.md](member1/README.md) for source code, road graphs, audited candidate tables, and reproduction instructions.

## Application and model integration

Member 4's FastAPI and Streamlit application is in [member4/](member4/README.md). Chengapa K S integrated live candidate features, the selected Extra Trees classifier and Member 3's directed decoder into that application. FastAPI and Streamlit run on separate computers. Set the backend data root and point the frontend at the backend laptop's LAN address as described in the application guide.

The sixth classifier, Random Forest, and the six-model comparison are in [Member2/results_random_forest/](Member2/results_random_forest/README.md). Extra Trees remains the serving model under the mean grouped-CV Top-1 selection rule.
