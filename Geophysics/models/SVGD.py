import tensorflow as tf
class SVGD():
    def __init__(self,joint_log_post,num_particles = 250,num_iter=1000, dtype=tf.float32):
        self.dtype = dtype
        self.num_particles = num_particles
        self.num_latent = 2
        self.lr = 0.003
        self.alpha = .9
        self.fudge_factor = 1e-6
        self.num_iter = num_iter
        self.range_limit = [-3, 3]
        self.npoints_plot = 50
        self.joint_log_post = joint_log_post


    def get_median(self,v):
        v = tf.reshape(v, [-1])
        m = v.get_shape()[0]//2
        return tf.nn.top_k(v, m).values[m-1]

    def svgd_kernel(self,X0):
        XY = tf.matmul(X0, tf.transpose(X0))
        X2_ = tf.reduce_sum(tf.square(X0), axis=1)

        x2 = tf.reshape(X2_, shape=(tf.shape(X0)[0], 1))

        X2e = tf.tile(x2, [1, tf.shape(X0)[0]])
        
        ## (x1 -x2)^2 + (y1 -y2)^2
        H = tf.subtract(tf.add(X2e, tf.transpose(X2e)), 2 * XY)

        V = tf.reshape(H, [-1, 1])

        # median distance

        h = self.get_median(V)
        h = tf.sqrt(
            0.5 * h / tf.math.log(tf.cast(tf.shape(X0)[0], self.dtype) + 1.0))

        # compute the rbf kernel
        Kxy = tf.exp(-H / h ** 2 / 2.0)

        dxkxy = tf.negative(tf.matmul(Kxy, X0))
        sumkxy = tf.expand_dims(tf.reduce_sum(Kxy, axis=1), 1)
        dxkxy = tf.add(dxkxy, tf.multiply(X0, sumkxy)) / (h ** 2)

        return (Kxy, dxkxy)


    def gradient(self,mu):
        log_p_grad = tf.TensorArray(self.dtype, size=self.num_particles)
        for i in range(mu.shape[0]):
            with tf.GradientTape() as t:
                t.watch(mu)
                f = self.joint_log_post(mu[i])
            log_p_grad =log_p_grad.write(i, t.gradient(f,mu)[i])
        return log_p_grad.stack()


    def svgd_one_iter(self,mu):
        # mu_norm = self.normalizer.encode(mu)
        log_p_grad = self.gradient(mu)
        kernel_matrix, kernel_gradients = self.svgd_kernel(mu)
        grad_theta = (tf.matmul(kernel_matrix, log_p_grad) + kernel_gradients) / self.num_particles
        # print(grad_theta)
        # mu_norm = mu_norm + self.lr * grad_theta
        mu = mu + self.lr * grad_theta
        # mu = self.normalizer.decode(mu_norm)
        # GPU = GPUInfo.gpu_usage()
        
        # print('GPU usage: {} %, GPU Memory: {} Mb'.format(GPU[0][0],GPU[1][0]))
        return mu

    def run_chain_svgd(self, mu):
        mu_list = []
        for i in range(self.num_iter):
            mu = self.svgd_one_iter(mu)
            if i // 10 == 0:
              print('step {}'.format(i))
            mu_list.append(mu.numpy())
        return mu,mu_list
