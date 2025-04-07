svc = pickle.load(open('data/svc.pkl', 'rb'))
rf_model = pickle.load(open('models/random_forest_text_model.pkl', 'rb'))
dt_model = pickle.load(open('models/decision_tree_text_model.pkl', 'rb'))
nb_model = joblib.load('models/naive_bayes_text_model.pkl')