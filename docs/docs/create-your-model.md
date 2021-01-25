# **為手刻的演算法建立模型類別**

我們只要簡單設定`__init__`、`fit`、`predict`，便可以建立為演算法建立起完整、可應用的模型類別。

### **1. 建立類別與 `__init__`函式**

```python
class ExampleClassifier(Model):

    model_type = "binary-classifier"

    def __init__(self, optimizer, metrics, **kwargs):
        self.weight_ = None
        self.compile(optimizer=optimizer, metrics=metrics)
        self.fit(**kwargs)

    #...
```

開始的第一步，我們透過繼承`base.Model`產生自訂模型的類別。

接著二個必要步驟：

* 建立類別變數 `model_type`:
    * 如果模型的輸出項是連續數值，填入`"regressor"`
    * 模型輸出為離散的類別資料，若為二元分類設定為`"binary-classifier"`
    * 多類別分類則設定`"multiclass-classifier"`
* 建立`__init__`:
    * 我們遵守scikit-learn的變數命名規則，設定以`_`結尾命名的變數作為模型訓練結果(範例中為`self.weight_`)
    * 設定優化器`optimizier`和模型度量`metrics`，與其他模型參數：
        利用`self.compile`將外部參數指定給模型內部的欄位，
        如此一來，我們就可以透過別名字串作為輸入，而不需要傳入`Optimizer`或`Metric`的實體
    * <del>一鍵訓練:`self.fit(**kwargs)`</del>

模型也可以寫定`Optimizer`和`metric`，僅提供優化器的參數供使用者調整：

```python
class ExampleClassifier(Model):

    model_type = "binary-classifier"

    def __init__(self, lr, **kwargs):
        self.weight_ = None
        self.compile(
            optimizer = ExampleOptimizer(lr=lr), 
            metrics = ['mae', 'acc']
        )
    
    # ...

```

### **2. 建立 `fit`**

```python
class ExampleClassifier(Model):

    # ... 

    @fit_method
    def fit(self, X, y, **kwargs):
        self.weight_ = self.optimzier.execute(X, y, **kwargs)
        return self
    
    # ...
```

* 建議使用`fit`命名模型的訓練函式，並以`X`、`y`分別代表訓練特徵及樣本值。
* 在訓練函式定義前加上`@fit_method`，當模型為二元分類時，資料會被預先編碼，再轉換為 (-1, 1) 的數值。
因此演算法實作僅需考慮(-1, 1)的二值輸入即可。

### **3. 建立 `predict`**

```python

class ExampleClassifier(Model):
    # ...

    @predict_method
    def predict(self, X):
        return X @ self.weight_
    
    # ...
```

* 建議使用`predict`命名模型的預測函式。
* 在預測函式定義前加上`@predict_method`，當模型為二元分類時，(-1, 1)的預測值會根據編碼轉換為原本的類別值輸出。
因此演算法實作僅需傳出(-1, 1)即可。


### **小結**

以下是模型類別完成的結果。別忘了預先載入`Model`類別和`fit_method`、`predict_method`裝飾器。

```python
from mlforge.base.models import Model
from mlforge.utils import fit_method, predict_method

class ExampleClassifier(Model):
    
    model_type = "classifier" 

    def __init__(self, optimizer, metrics, **kwargs):
        self.weight_ = None                        
        
        self.optimizer = optimizer
        self.metrics = metrics

    @fit_method
    def fit(self, X, y, **kwargs):
        self.weight_ = self.optimzier.execute(X, y, **kwargs)
        return self

    @predict_method
    def predict(self, X):
        return X @ self.weight_
```