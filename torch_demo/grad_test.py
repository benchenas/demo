import paddle

x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
y = paddle.to_tensor([4.0, 5.0, 6.0], stop_gradient=False)
z = x ** 2 + 4 * y

z.backward()
print("Tensor x's grad is: {}".format(x.grad))
print("Tensor y's grad is: {}".format(y.grad))