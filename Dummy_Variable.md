
```python
#Cria variável Dummy
 
dummy_est_civ = pd.get_dummies(df['est_civ'], prefix='est_civ', drop_first=True)
dummy_est_civ = dummy_est_civ.astype(int)
print(dummy_est_civ)
```
