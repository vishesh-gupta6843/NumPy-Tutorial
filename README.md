# NumPy-Tutorial

This repository is for keeping a track of my **NumPy learning journey** and implementing functions in real-world scenarios, especially within the context of **Data Science** and **Machine Learning**.

---

## 📚 Topics to Learn

The journey is structured from foundational NumPy concepts to advanced techniques and real-world applications.

### 1. Basics (Foundational)
- Array attributes: `.shape`, `.ndim`, `.dtype`, `.size`, `.itemsize`, `.nbytes`
- Array creation: `np.array()`, `np.arange()`, `np.linspace()`, `np.zeros()`, `np.ones()`, `np.full()`, `np.eye()`
- Data types and conversions: `.astype()`, `int32`, `float64`, `bool`, `object`

### 2. Indexing, Slicing & Manipulation
- Indexing & slicing: `a[3]`, `a[1:5]`, `a[1, 2]`, `a[:, 0]`, `a[1, :]`
- Boolean and fancy indexing: `a[a > 5]`, `a[[0, 2, 4]]`
- Array reshaping: `reshape()`, `ravel()`, `flatten()`, `transpose()`, `swapaxes()`, `moveaxis()`, `expand_dims()`, `squeeze()`, `concatenate()`, `stack()`, `vstack()`, `hstack()`, `split()`

### 3. Math & Statistics
- Element-wise operations: `+`, `-`, `*`, `/`, `**`, `np.sqrt()`, `np.exp()`, `np.log()`
- Logical and comparison operators
- Aggregation: `np.sum()`, `np.mean()`, `np.median()`, `np.std()`, `np.var()`, `np.min()`, `np.max()`
- Use of `axis` parameter
- Index functions: `np.argmin()`, `np.argmax()`, `np.where()`, `np.nonzero()`
- Cumulative functions: `np.cumsum()`, `np.cumprod()`, `np.diff()`

### 4. Broadcasting & Shape Compatibility
- Broadcasting rules
- Use of `np.newaxis`, `reshape()`
- Use cases: scalar + array, column + matrix, row + matrix

### 5. Linear Algebra
- Matrix operations: `@`, `np.transpose()`, `np.linalg.inv()`, `np.linalg.det()`, `np.linalg.eig()`, `np.trace()`, `np.linalg.solve()`

### 6. Handling Missing Data (NaNs)
- Checking: `np.isnan()`, `np.isinf()`
- Replacing: `np.nan_to_num()`, `np.where(np.isnan(a), fill_value, a)`
- Aggregation with NaNs: `np.nanmean()`, `np.nansum()`

### 7. Random Number Generation
- Random functions: `np.random.rand()`, `np.random.uniform()`, `np.random.randn()`, `np.random.normal()`, `np.random.randint()`
- Seeding: `np.random.seed()`
- Shuffling: `np.random.shuffle()`, `np.random.permutation()`

### 8. Sorting, Searching & Set Operations
- Sorting: `np.sort()`, `np.argsort()`, `np.searchsorted()`
- Set operations: `np.unique()`, `np.union1d()`, `np.intersect1d()`, `np.setdiff1d()`

### 9. Performance & Vectorization
- Vectorized operations
- Functions: `np.vectorize()`, `np.apply_along_axis()`
- Understanding views vs copies

### 10. Interfacing with Other Libraries
- Pandas: `pd.DataFrame(np_array)`, `.values`
- Scikit-learn: model input/output compatibility
- TensorFlow/PyTorch: tensor conversions

### Bonus: Utility Functions
- Math helpers: `np.clip()`, `np.round()`, `np.floor()`, `np.ceil()`
- Logical helpers: `np.all()`, `np.any()`
- File I/O: `np.save()`, `np.load()`, `np.savetxt()`, `np.genfromtxt()`

---

## 💡 Real-World Applications (To Be Implemented)

- Data cleaning: filtering, NaN handling
- Feature scaling: normalization and standardization
- Matrix algebra: linear regression, PCA
- Vectorized predictions for batch inputs

---

## 🛠️ Folder Structure (Planned)
- `topics.txt` – Progress checklist
- `.ipynb` notebooks and `.py` scripts – Concept implementation
- `real_world/` – Applied examples and mini-projects

---

## 🚀 Goal

To build a solid understanding of NumPy from scratch and apply it efficiently in real-world data workflows and ML pipelines.

---

## 🙌 Contributions

This is a personal learning tracker, but I welcome ideas, corrections, or resources that can improve the learning experience.



