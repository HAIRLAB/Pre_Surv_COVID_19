import pickle
model = pickle.load(open("single-tree.dat", "rb"))
x_demo = np.array([[368,20,3], [300,100,10]]) #'乳酸脱氢酶', '淋巴细胞(%)', '超敏C反应蛋白'
y_demo_pred = model.predict(x_demo)
y_demo_proba = model.predict_proba(x_demo)
