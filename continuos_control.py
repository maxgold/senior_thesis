import numpy as np
import tensorflow as tf
import gym
import scipy.signal

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NnValueFunction(object):
    def __init__(self, ob_dim, session, n_epochs, stepsize):
        self._out= None
        self._ob_dim = ob_dim
        self._session = session
        self._n_epochs = n_epochs
        self._stepsize = stepsize

    def _init_net(self, shape):
        self._x = tf.placeholder(shape=[None, shape], name="NnValue_x", dtype=tf.float32)
        self._y = tf.placeholder(shape=[None], name="NnValue_y", dtype=tf.float32)
        hidden1 = lrelu(dense(self._x, 32, "NnValue_h1", weight_init=normc_initializer(1.0)))
        hidden2 = lrelu(dense(hidden1, 16, "NnValue_h2", weight_init=normc_initializer(1.0)))
        self._out = dense(hidden2, 1, "NnValue_output", weight_init=normc_initializer(1.0))

        # mean squared error
        loss = tf.reduce_mean(tf.pow(self._y - self._out, 2))
        self._train_op = tf.train.AdamOptimizer(self._stepsize).minimize(loss)
        self._session.run(tf.global_variables_initializer())
    def predict(self, X):
        if self._out is None:
            return np.zeros(X.shape[0])
        else:
            out = self._session.run(self._out, feed_dict={self._x: self.preproc(X)})
            return np.squeeze(out)
    def fit(self, X, Y):
        X = self.preproc(X)
        if self._out is None:
            self._init_net(X.shape[1])
        for _ in range(self._n_epochs):
          self._session.run(self._train_op, feed_dict={self._x: X, self._y: Y})
    # Keep this preproc to make the comparison with LinearValueFunction
    # more clear.
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_type, vf_params, animate=False):
    seed = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0))) # hidden layer
    sy_h2 = lrelu(dense(sy_h1, 16, "h2", weight_init=normc_initializer(1.0))) # hidden layer
    # Gaussian distribution (mean, stdev) for each action dimension for the
    # batch.
    sy_mean_na = dense(sy_h2, ac_dim, "mean", weight_init=normc_initializer(0.05))
    # Use the same stdev for all inputs.
    sy_logstd_a = tf.get_variable("logstdev", [ac_dim], initializer=tf.zeros_initializer()) # Variance

    sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) # batch of actions taken by the policy, used for policy gradient computation

    # Now, need to compute the logprob for each action taken.
    action_dist = tf.contrib.distributions.Normal(loc=sy_mean_na, scale=tf.exp(sy_logstd_a), validate_args=True)
    # sy_logprob_n is in [batch_size, ac_dim] shape.
    sy_logprob_n = action_dist.log_prob(sy_ac_na)

    # Now, need to sample an action based on input. This should be a 1-D vector
    # with ac_dim float in it.
    sy_sampled_ac = action_dist.sample()[0]

    # old mean/stdev before updating the policy. This is purely used for
    # computing KL
    sy_oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_oldlogstd_a = tf.placeholder(shape=[ac_dim], name='oldlogstdev', dtype=tf.float32)
    old_action_dist = tf.contrib.distributions.Normal(loc=sy_oldmean_na, scale=tf.exp(sy_oldlogstd_a), validate_args=True)
    sy_kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(action_dist, old_action_dist))
    # Compute entropy
    sy_ent = tf.reduce_mean(action_dist.entropy())

    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)  # advantage function estimate

    # We do tf.reduce_mean on sy_logprob_n here, as it's shape is [batch_size,
    # ac_dim]. Not sure what's the best way to deal with ac_dim -- but pendulum's
    # ac_dim is 1, so using reduce_mean here is fine.
    sy_surr = - tf.reduce_mean(sy_adv_n * tf.reduce_mean(sy_logprob_n, 1)) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(ob_dim=ob_dim, session=sess, **vf_params)

    initial_ob = env.reset()

    total_timesteps = 0
    stepsize = initial_stepsize

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, oldmean_na, oldlogstd_a = sess.run([update_op, sy_mean_na, sy_logstd_a], feed_dict={sy_ob_no:ob_no, sy_ac_na:ac_na, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldmean_na: oldmean_na, sy_oldlogstd_a: oldlogstd_a})

        if kl > desired_kl * 2:
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2:
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')


if __name__ == "__main__":
    if 0:
        main_cartpole(logdir=None) # when you want to start collecting results, set the logdir
    if 1:
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=300, initial_stepsize=1e-3)
        params = [
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        ]
        import multiprocessing
        p = multiprocessing.Pool()
        p.map(main_pendulum1, params)