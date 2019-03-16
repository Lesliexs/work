import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
df = pd.read_csv('data/ex1data1.txt',names=['population','profit'])
# print(df.head(5))
# df.info()

# sns.lmplot将数据创建回归模型
sns.lmplot('population','profit',df,height=6,fit_reg=False)
# df.plot()
# plt.show()


# print(len(df))
# 获得样本特征
def get_X(df):
    ones = pd.DataFrame({'ones':np.ones(len(df))})
    data = pd.concat([ones,df],axis=1)

    return data.iloc[:,:-1].values
# 获取最后一列的标签数据
def get_y(df):
    return np.array(df.iloc[:,-1])

# 均值归一化 apply函数将column每一列数据依次传入 lambda函数将column的值传给后面的表达式
def normalize_feature(df):
    return df.apply(lambda column:(column-column.mean())/column.std())

nf = normalize_feature(df)
print(nf)
data = pd.read_csv('data/ex1data1.txt',names=['population','profit'])
print(data.head(5))

X = get_X(data)
print(X)
print(X.shape)

y = get_y(data)
print(y.shape)

theta = np.zeros(X.shape[1])
print(theta)

m = X.shape[0]
# inner = np.power(((X.dot(theta))-y),2)
# print(inner)
# inner = np.power(((X*theta.T)-y),2)
# print(inner)
# 创建损失函数
def cost(theta,X,y):
    inner = np.power(((X.dot(theta.T))- y),2)
    # print(X.dot(theta))
    # print(inner)
    # print(inner.shape)
    square_sum = sum(inner)
    # costf = square_sum/(2*m)
    # return costf
    return np.sum(inner)/(2*m)

a = cost(theta,X,y)
# print(a)

def gradient(theta,X,y):
    m = X.shape[0]
    inner = X.T.dot(X.dot(theta)-y)
    return inner/m

def batch_gradient_decnet(theta,X,y,epoch,alpha=0.01):
    cost_data = [cost(theta,X,y)]
    _theta = theta.copy()
    for _ in range(epoch):
        _theta = _theta-alpha*gradient(_theta,X,y)
        cost_data.append(cost(_theta,X,y))
    return _theta,cost_data

epoch = 500
final_thata,cost_data = batch_gradient_decnet(theta,X,y,epoch)
print(final_thata)
print(cost_data)

print(cost(final_thata,X,y))

ax = sns.tsplot(cost_data,time=np.arange(epoch+1))
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.show()

b = final_thata[0]
m = final_thata[1]
plt.scatter(data.population,data.profit,label='Training data')
plt.plot(data.population,data.population*m+b,label = 'Prediction')
plt.legend(loc=2)
plt.show()