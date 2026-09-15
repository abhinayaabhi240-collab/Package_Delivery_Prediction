import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/package_delivery_data.csv")
X = df.drop("delivery_time_hours", axis=1)
y = df["delivery_time_hours"]
cat = ["traffic_level","weather","vehicle_type","priority"]
num = ["distance_km","package_weight_kg","warehouse_delay_hours","handling_time_hours"]
pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num","passthrough",num)])
pipe = Pipeline([("preprocessor",pre),("model",RandomForestRegressor(n_estimators=200,random_state=42))])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
pipe.fit(X_train,y_train)
p=pipe.predict(X_test)
print("MAE:",round(mean_absolute_error(y_test,p),2))
print("RMSE:",round(mean_squared_error(y_test,p)**.5,2))
print("R2:",round(r2_score(y_test,p),4))
joblib.dump(pipe,"package_delivery_model.pkl")
names=pipe.named_steps["preprocessor"].get_feature_names_out()
imp=pipe.named_steps["model"].feature_importances_
d=pd.DataFrame({"feature":names,"importance":imp}).sort_values("importance",ascending=False).head(15)
plt.figure(figsize=(10,6)); plt.barh(d["feature"][::-1],d["importance"][::-1]); plt.tight_layout(); plt.savefig("feature_importance.png")
