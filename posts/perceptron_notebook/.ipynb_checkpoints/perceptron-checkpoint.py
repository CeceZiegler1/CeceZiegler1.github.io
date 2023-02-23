{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4e1314ed-de8a-4db8-a56a-5303fe21304e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "#X_ = np.append(X, np.ones((X.shape[0], 1)), 1)\n",
    "\n",
    "class Perceptron():\n",
    "    \n",
    "    def __init__(self):\n",
    "        self.w\n",
    "        self.bias\n",
    "        self.history\n",
    "    \n",
    "    def fit(self, X, y, max_steps):\n",
    "        w = np.random.random(X.shape)#Want the p from the Xnbyp matrix +1 to the p so that it is w tild/ w-b save it the instance variable\n",
    "        self.w = w\n",
    "        while loss != 0:\n",
    "            for i in range(max_steps):\n",
    "                dp = np.dot(X[i,:], w) #computing dot product\n",
    "                y_predictor = np.sign(dp) #computing sign of the predictor value\n",
    "                self.weights += 1*(y_predictor < 0)*y[i]*X[i,:] #updating weights\n",
    "            loss = 1-self.accuracy(X, y)\n",
    "            \n",
    "                \n",
    "    def score(self, X, y):\n",
    "        return((np.dot(X, self.w)*y) > 0).mean()\n",
    "    \n",
    "    def predict(self, X):\n",
    "        pred = np.dot(X, self.w)\n",
    "        \n",
    "        pred[pred > 0] = 1\n",
    "        pred[pred <= 0] = 0\n",
    "        \n",
    "        return pred\n",
    "    \n",
    "                \n",
    "        \n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c5ed101-c0dd-40c7-9022-4c4d0c83c6c4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07dd456c-63f5-41b4-b921-66f3d8f7da90",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:ml-0451] *",
   "language": "python",
   "name": "conda-env-ml-0451-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
