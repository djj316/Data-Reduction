# ���ݹ�Լ����ʵ�鱨�棺���� Wine Quality ���ݼ��� PCA �� LDA ����

**GitHub �ֿ�**��[(https://github.com/djj316/zyh.)]  
**������**��2025��4��13��  

---

## Ŀ¼
1. [ʵ��Ŀ��](#1-ʵ��Ŀ��)  
2. [���ݼ�](#2-���ݼ�)  
3. [����](#3-����)  
4. [���](#4-���)  
5. [����](#5-����)  
6. [����](#6-����)  
7. [��¼](#7-��¼)  

---

## 1. ʵ��Ŀ��
�Ƚ� PCA�����ɷַ������� LDA�������б�����������Ѿ��������������еı��֣�
- ������ͬ��ά�����Է���׼ȷ�ʵ�Ӱ��  
- ���ӻ���ά�����ڵ�ά�ռ��ͶӰ�ṹ  
- �����������ά��������Ȩ���ϵ  

---

## 2. ���ݼ�
### ������Դ
UCI Machine Learning Repository �� [Wine Quality Dataset (ID:186)](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

### ����˵��
| ��������       | ���� | ʾ��                  |
|----------------|------|-----------------------|
| ��ָ��       | 11   | ��ȡ�pHֵ���ƾ�Ũ�ȵ� |
| Ŀ�����       | 1    | �������֣�3-9��       |

### Ԥ����
```python
# �������ַ��䣨3�ࣩ
y = np.digitize(y, bins=[3, 6], right=True) - 1
# ���ݻ��֣�70%ѵ������30%���Լ���
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## 3. ����
### ��������
```mermaid
graph LR
A[ԭʼ����] --> B[��׼��]
B --> C[PCA��ά]
B --> D[LDA��ά]
C --> E[KNN����ģ��1]
D --> F[KNN����ģ��2]
B --> G[KNN����ģ��3]
```

### �ؼ�����
#### ѡ������ʵ�PCAά��
```python
pca_full = PCA()
pca_full.fit(X_train_scaled)
n_components_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
```

#### LDA ͶӰ
```python
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
```
---

## 4. ���
### 4.1 ����׼ȷ�ʶԱ�
| ����       | ����׼ȷ�� | ��ά��ά�� |
|------------|------------|------------|
| ԭʼ����    | 0.80       | 11         |
| PCA        | 0.84       | 9          |
| LDA        | 0.82       | 2          |

### 4.2 ���ӻ�
#### PCA �ۼƷ��������
![PCA Cumulative Variance](./PCA�ۼƽ��ͷ���ͼ%20.png)  
*����95%������Ҫ9�����ɷ�*

#### LDA vs PCA ��άͶӰ
![LDA vs PCA](./LDA��PCAͶӰ���ӻ�%20.png)  

#### ģ�ͷ���׼ȷ�ʱȶ�
![LDA vs PCA vs Common](./ģ��׼ȷ�ʶԱ�.png) 

---

## 5. ����
### PCA ����
- ǰ�������ɷֽ�����50%����  
- **�ŵ�**���޼ල�������ʺ�����̽��  
- **ȱ��**����άͶӰ��ʧ���ַ�����Ϣ  

### LDA ����
- ǿ�ƽ�ά��2ά�Ա��ֽϸ߷�������  
- **�ŵ�**���ලѧϰ��ֱ���Ż�������  
- **ȱ��**����ཱུ��`C-1`ά��CΪ�������  

---

## 6. ����
1. PCA �ڱ���95%����ʱ�ɼ���18%ά��(11��9) 
2. LDA �ڼ��˽�ά(11��2)������Ч�ʸ���  
3. **�Ƽ�����**��  
   - ����ʹ��PCA���г�����������  
   - ����ǿ��ά�ұ�ǩ�ɿ���ѡ��LDA  
---

## 7. ��¼
### ��������
```bash
Python 3.8+  
��������  
- numpy>=1.21  
- scikit-learn>=1.0  
- matplotlib>=3.5  
```

### ����˵��
1. ��¡�ֿ⣺
   ```bash
   git clone [���Ĳֿ�URL]
   ```
2. ��װ������
   ```bash
   pip install -r requirements.txt
   ```
3. ִ��ʵ�飺
   ```bash
   python scripts/wine_analysis.py
   ```

---

**��������**��zyh 
**License**��MIT

