"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com

Credits: 
  1.  Jonathan Hui's blog, "Understanding Matrix capsules with EM Routing 
      (Based on Hinton's Capsule Networks)" 
      https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-
      Capsule-Network/
  2.  Questions and answers on OpenReview, "Matrix capsules with EM routing" 
      https://openreview.net/forum?id=HJWLfGWRb
  3.  Suofei Zhang's implementation on GitHub, "Matrix-Capsules-EM-Tensorflow" 
      https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
  4.  Guang Yang's implementation on GitHub, "CapsulesEM" 
      https://github.com/gyang274/capsulesEM
"""

# Public modules
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def em_routing(votes_ij, activations_i, batch_size, spatial_routing_matrix):
  """The EM routing between input capsules (i) and output capsules (j).
  
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of EM routing.
  
  Author:
    Ashley Gritzman 19/10/2018
  Definitions:
    N -> number of samples in batch
    OH -> output height
    OW -> output width
    kh -> kernel height
    kw -> kernel width
    kk -> kh * kw
    i -> number of input capsules, also called "child_caps"
    o -> number of output capsules, also called "parent_caps"
    child_space -> spatial dimensions of input capsule layer i
    parent_space -> spatial dimensions of output capsule layer j
    n_channels -> number of channels in pose matrix (usually 4x4=16)
  Args: 
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
      For conv layer:
        (N*OH*OW, kh*kw*i, o, 4x4)
        (64*6*6, 9*8, 32, 16)
      For FC layer:
        The kernel dimensions are equal to the spatial dimensions of the input 
        layer i, and the spatial dimensions of the output layer j are 1x1.
        (N*1*1, child_space*child_space*i, o, 4x4)
        (64, 4*4*16, 5, 16)
    activations_i: 
      activations of capsules in layer i (L)
      (N*OH*OW, kh*kw*i, 1)
      (64*6*6, 9*8, 1)
    batch_size: 
    spatial_routing_matrix: 
  Returns:
    poses_j: 
      poses of capsules in layer j (L+1)
      (N, OH, OW, o, 4x4) 
      (64, 6, 6, 32, 16)
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, OH, OW, o, 1)
      (64, 6, 6, 32, 1)
  """
  
  #----- Dimensions -----#
  
  # Get dimensions needed to do conversions
  N = batch_size
  votes_shape = votes_ij.get_shape().as_list()
  OH = np.sqrt(int(votes_shape[0]) / N)
  OH = int(OH)
  OW = np.sqrt(int(votes_shape[0]) / N)
  OW = int(OW)
  kh_kw_i = int(votes_shape[1])
  o = int(votes_shape[2])
  n_channels = int(votes_shape[3])
  
  # Calculate kernel size by adding up column of spatial routing matrix
  # Do this before conventing the spatial_routing_matrix to tf
  kk = int(np.sum(spatial_routing_matrix[:,0]))
  
  parent_caps = o
  child_caps = int(kh_kw_i/kk)
  
  rt_mat_shape = spatial_routing_matrix.shape
  child_space_2 = rt_mat_shape[0]
  child_space = int(np.sqrt(child_space_2))
  parent_space_2 = rt_mat_shape[1]
  parent_space = int(np.sqrt(parent_space_2))
   
  
  #----- Reshape Inputs -----#

  # conv: (N*OH*OW, kh*kw*i, o, 4x4) -> (N, OH, OW, kh*kw*i, o, 4x4)
  # FC: (N, child_space*child_space*i, o, 4x4) -> (N, 1, 1, child_space*child_space*i, output_classes, 4x4)
  votes_ij = tf.reshape(votes_ij, [N, OH, OW, kh_kw_i, o, n_channels]) 
  
  # (N*OH*OW, kh*kw*i, 1) -> (N, OH, OW, kh*kw*i, o, n_channels)
  #              (24, 6, 6, 288, 1, 1)
  activations_i = tf.reshape(activations_i, [N, OH, OW, kh_kw_i, 1, 1])
  

  #----- Betas -----#

  """
  # Initialization from Jonathan Hui [1]:
  beta_v_hui = tf.get_variable(
    name='beta_v', 
    shape=[1, 1, 1, o], 
    dtype=tf.float32,
    initializer=tf.contrib.layers.xavier_initializer())
  beta_a_hui = tf.get_variable(
    name='beta_a', 
    shape=[1, 1, 1, o], 
    dtype=tf.float32,
    initializer=tf.contrib.layers.xavier_initializer())
                              
  # AG 21/11/2018: 
  # Tried to find std according to Hinton's comments on OpenReview 
  # https://openreview.net/forum?id=HJWLfGWRb&noteId=r1lQjCAChm
  # Hinton: "We used truncated_normal_initializer and set the std so that at the 
  # start of training half of the capsules in each layer are active and half 
  # inactive (for the Primary Capsule layer where the activation is not computed 
  # through routing we use different std for activation convolution weights & 
  # for pose parameter convolution weights)."
  # 
  # std beta_v seems to control the spread of activations
  # To try and achieve what Hinton said about half active and half not active,
  # I change the std values and check the histogram/distributions in 
  # Tensorboard
  # to try and get a good spread across all values. I couldn't get this working
  # nicely.
  beta_v_hui = slim.model_variable(
    name='beta_v', 
    shape=[1, 1, 1, 1, o, 1], 
    dtype=tf.float32,
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=10.0))
  """
  beta_a = slim.model_variable(
    name='beta_a', 
    shape=[1, 1, 1, 1, o, 1], 
    dtype=tf.float32, 
    initializer=tf.truncated_normal_initializer(mean=-1000.0, stddev=500.0))
  
  # AG 04/10/2018: using slim.variable to create instead of tf.get_variable so 
  # that they get correctly placed on the CPU instead of GPU in the multi-gpu 
  # version.
  # One beta per output capsule type
  # (1, 1, 1, 1, 32, 1)
  # (N, OH, OH, i, o, n_channels)
  beta_v = slim.model_variable(
    name='beta_v', 
    shape=[1, 1, 1, 1, o, 1], 
    dtype=tf.float32,            
    initializer=tf.contrib.layers.xavier_initializer(),
    regularizer=None)
  """
  beta_a = slim.model_variable(
    name='beta_a', 
    shape=[1, 1, 1, 1, o, 1], 
    dtype=tf.float32, 
    initializer=tf.contrib.layers.xavier_initializer(),
    regularizer=None)
  """

  with tf.variable_scope("em_routing") as scope:
    # Initialise routing assignments
    # rr (1, 6, 6, 9, 8, 16) 
    #  (1, parent_space, parent_space, kk, child_caps, parent_caps)
    rr = init_rr(spatial_routing_matrix, child_caps, parent_caps)
    
    # Need to reshape (1, 6, 6, 9, 8, 16) -> (1, 6, 6, 9*8, 16, 1)
    rr = np.reshape(
      rr, 
      [1, parent_space, parent_space, kk*child_caps, parent_caps, 1])
    
    # Convert rr from np to tf
    rr = tf.constant(rr, dtype=tf.float32)
    
    for it in range(FLAGS.iter_routing):  
      # AG 17/09/2018: modified schedule for inverse_temperature (lambda) based
      # on Hinton's response to questions on OpenReview.net: 
      # https://openreview.net/forum?id=HJWLfGWRb
      # "the formula we used for lambda is:
      # lambda = final_lambda * (1 - tf.pow(0.95, tf.cast(i + 1, tf.float32)))
      # where 'i' is the routing iteration (range is 0-2). Final_lambda is set 
      # to 0.01."
      # final_lambda = 0.01
      final_lambda = FLAGS.final_lambda
      inverse_temperature = (final_lambda * 
                             (1 - tf.pow(0.95, tf.cast(it + 1, tf.float32))))

      # AG 26/06/2018: added var_j
      activations_j, mean_j, stdv_j, var_j = m_step(
        rr, 
        votes_ij, 
        activations_i, 
        beta_v, beta_a, 
        inverse_temperature=inverse_temperature)
      
      # We skip the e_step call in the last iteration because we only need to 
      # return the a_j and the mean from the m_stp in the last iteration to 
      # compute the output capsule activation and pose matrices  
      if it < FLAGS.iter_routing - 1:
        rr = e_step(votes_ij, 
                    activations_j, 
                    mean_j, 
                    stdv_j, 
                    var_j, 
                    spatial_routing_matrix)

    # pose: (N, OH, OW, o, 4 x 4) via squeeze mean_j (24, 6, 6, 32, 16)
    poses_j = tf.squeeze(mean_j, axis=-3, name="poses")

    # activation: (N, OH, OW, o, 1) via squeeze o_activation is 
    # [24, 6, 6, 32, 1]
    activations_j = tf.squeeze(activations_j, axis=-3, name="activations")

  return poses_j, activations_j


def m_step(rr, votes, activations_i, beta_v, beta_a, inverse_temperature):
  """The m-step in EM routing between input capsules (i) and output capsules 
  (j).
  
  Compute the activations of the output capsules (j), and the Gaussians for the
  pose of the output capsules (j).
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of m-step.
  
  Author:
    Ashley Gritzman 19/10/2018
    
  Args: 
    rr: 
      assignment weights between capsules in layer i and layer j
      (N, OH, OW, kh*kw*i, o, 1)
      (64, 6, 6, 9*8, 16, 1)
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
      For conv layer:
        (N, OH, OW, kh*kw*i, o, 4x4)
        (64, 6, 6, 9*8, 32, 16)
      For FC layer:
        The kernel dimensions are equal to the spatial dimensions of the input 
        layer i, and
        the spatial dimensions of the output layer j are 1x1.
        (N, 1, 1, child_space*child_space*i, output_classes, 4x4)
        (64, 1, 1, 4*4*16, 5, 16)
    activations_i: 
      activations of capsules in layer i (L)
      (N, OH, OW, kh*kw*i, o, n_channels)
      (24, 6, 6, 288, 1, 1)
    beta_v: 
      Trainable parameters in computing cost 
      (1, 1, 1, 1, 32, 1)
    beta_a: 
      Trainable parameters in computing next level activation 
      (1, 1, 1, 1, 32, 1)
    inverse_temperature: lambda, increase over each iteration by the caller
    
  Returns:
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, OH, OW, 1, o, 1)
      (64, 6, 6, 1, 32, 1)
    mean_j: 
      mean of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    stdv_j: 
      standard deviation of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    var_j: 
      variance of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
  """

  with tf.variable_scope("m_step") as scope:
    
    rr_prime = rr * activations_i
    rr_prime = tf.identity(rr_prime, name="rr_prime")

    # rr_prime_sum: sum over all input capsule i
    rr_prime_sum = tf.reduce_sum(rr_prime, 
                                 axis=-3, 
                                 keepdims=True, 
                                 name='rr_prime_sum')
    
    # AG 13/12/2018: normalise amount of information
    # The amount of information given to parent capsules is very different for 
    # the final "class-caps" layer. Since all the spatial capsules give output 
    # to just a few class caps, they receive a lot more information than the 
    # convolutional layers. So in order for lambda and beta_v/beta_a settings to 
    # apply to this layer, we must normalise the amount of information.
    # activ from convcaps1 to convcaps2 (64*5*5, 144, 16, 1) 144/16 = 9 info
    # (N*OH*OW, kh*kw*i, o, 1)
    # activ from convcaps2 to classcaps (64, 1, 1, 400, 5, 1) 400/5 = 80 info
    # (N, 1, 1, IH*IW*i, n_classes, 1)
    child_caps = float(rr_prime.get_shape().as_list()[-3])
    parent_caps = float(rr_prime.get_shape().as_list()[-2])
    ratio_child_to_parent =  child_caps/parent_caps
    layer_norm_factor = 100/ratio_child_to_parent
    # logger.info("ratio_child_to_parent: {}".format(ratio_child_to_parent))
    # rr_prime_sum = rr_prime_sum/ratio_child_to_parent

    # mean_j: (24, 6, 6, 1, 32, 16)
    mean_j_numerator = tf.reduce_sum(rr_prime * votes, 
                                     axis=-3, 
                                     keepdims=True, 
                                     name="mean_j_numerator")
    mean_j = tf.div(mean_j_numerator, 
                    rr_prime_sum + FLAGS.epsilon, 
                    name="mean_j")
    
    #----- AG 26/06/2018 START -----#
    # Use variance instead of standard deviation, because the sqrt seems to 
    # cause NaN gradients during backprop.
    # See original implementation from Suofei below
    var_j_numerator = tf.reduce_sum(rr_prime * tf.square(votes - mean_j), 
                                    axis=-3, 
                                    keepdims=True, 
                                    name="var_j_numerator")
    var_j = tf.div(var_j_numerator, 
                   rr_prime_sum + FLAGS.epsilon, 
                   name="var_j")
    
    # Set the minimum variance (note: variance should always be positive)
    # This should allow me to remove the FLAGS.epsilon safety from log and div 
    # that follow
    #var_j = tf.maximum(var_j, FLAGS.epsilon)
    #var_j = var_j + FLAGS.epsilon
    
    ###################
    #var_j = var_j + 1e-5
    var_j = tf.identity(var_j + 1e-9, name="var_j_epsilon")
    ###################
    
    # Compute the stdv, but it shouldn't actually be used anywhere
    # stdv_j = tf.sqrt(var_j)
    stdv_j = None
    
    ######## layer_norm_factor
    cost_j_h = (beta_v + 0.5*tf.log(var_j)) * rr_prime_sum * layer_norm_factor
    cost_j_h = tf.identity(cost_j_h, name="cost_j_h")
    
    # ----- END ----- #
    
    """
    # Original from Suofei (reference [3] at top)
    # stdv_j: (24, 6, 6, 1, 32, 16)
    stdv_j = tf.sqrt(
      tf.reduce_sum(
        rr_prime * tf.square(votes - mean_j), axis=-3, keepdims=True
      ) / rr_prime_sum,
      name="stdv_j"
    )
    # cost_j_h: (24, 6, 6, 1, 32, 16)
    cost_j_h = (beta_v + tf.log(stdv_j + FLAGS.epsilon)) * rr_prime_sum
    """
    
    # cost_j: (24, 6, 6, 1, 32, 1)
    # activations_j_cost = (24, 6, 6, 1, 32, 1)
    # yg: This is done for numeric stability.
    # It is the relative variance between each channel determined which one 
    # should activate.
    cost_j = tf.reduce_sum(cost_j_h, axis=-1, keepdims=True, name="cost_j")
    #cost_j_mean = tf.reduce_mean(cost_j, axis=-2, keepdims=True)
    #cost_j_stdv = tf.sqrt(
    #  tf.reduce_sum(
    #    tf.square(cost_j - cost_j_mean), axis=-2, keepdims=True
    #  ) / cost_j.get_shape().as_list()[-2]
    #)
    
    # AG 17/09/2018: trying to remove normalisation
    # activations_j_cost = beta_a + (cost_j_mean - cost_j) / (cost_j_stdv)
    activations_j_cost = tf.identity(beta_a - cost_j, 
                                     name="activations_j_cost")

    # (24, 6, 6, 1, 32, 1)
    activations_j = tf.sigmoid(inverse_temperature * activations_j_cost,
                               name="sigmoid")
    
    # AG 26/06/2018: added var_j to return
    return activations_j, mean_j, stdv_j, var_j

  
# AG 26/06/2018: added var_j
def e_step(votes_ij, activations_j, mean_j, stdv_j, var_j, spatial_routing_matrix):
  """The e-step in EM routing between input capsules (i) and output capsules (j).
  
  Update the assignment weights using in routung. The output capsules (j) 
  compete for the input capsules (i).
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of e-step.
  
  Author:
    Ashley Gritzman 19/10/2018
    
  Args: 
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
      For conv layer:
        (N, OH, OW, kh*kw*i, o, 4x4)
        (64, 6, 6, 9*8, 32, 16)
      For FC layer:
        The kernel dimensions are equal to the spatial dimensions of the input 
        layer i, and the spatial dimensions of the output layer j are 1x1.
        (N, 1, 1, child_space*child_space*i, output_classes, 4x4)
        (64, 1, 1, 4*4*16, 5, 16)
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, OH, OW, 1, o, 1)
      (64, 6, 6, 1, 32, 1)
    mean_j: 
      mean of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    stdv_j: 
      standard deviation of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    var_j: 
      variance of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    spatial_routing_matrix: ???
    
  Returns:
    rr: 
      assignment weights between capsules in layer i and layer j
      (N, OH, OW, kh*kw*i, o, 1)
      (64, 6, 6, 9*8, 16, 1)
  """
  
  with tf.variable_scope("e_step") as scope:
    
    # AG 26/06/2018: changed stdv_j to var_j
    o_p_unit0 = - tf.reduce_sum(
      tf.square(votes_ij - mean_j, name="num") / (2 * var_j), 
      axis=-1, 
      keepdims=True, 
      name="o_p_unit0")
    
    o_p_unit2 = - 0.5 * tf.reduce_sum(
      tf.log(2*np.pi * var_j), 
      axis=-1, 
      keepdims=True, 
      name="o_p_unit2"
    )

    # (24, 6, 6, 288, 32, 1)
    o_p = o_p_unit0 + o_p_unit2
    zz = tf.log(activations_j + FLAGS.epsilon) + o_p
    
    # AG 13/11/2018: New implementation of normalising across parents
    #----- Start -----#
    zz_shape = zz.get_shape().as_list()
    batch_size = zz_shape[0]
    parent_space = zz_shape[1]
    kh_kw_i = zz_shape[3]
    parent_caps = zz_shape[4]
    kk = int(np.sum(spatial_routing_matrix[:,0]))
    child_caps = int(kh_kw_i / kk)
    
    zz = tf.reshape(zz, [batch_size, parent_space, parent_space, kk, 
                         child_caps, parent_caps])
    
    """
    # In un-log space
    with tf.variable_scope("to_sparse_unlog") as scope:
      zz_unlog = tf.exp(zz)
      #zz_sparse_unlog = utl.to_sparse(zz_unlog, spatial_routing_matrix, 
      # sparse_filler=1e-15)
      zz_sparse_unlog = utl.to_sparse(
          zz_unlog, 
          spatial_routing_matrix, 
          sparse_filler=0.0)
      # maybe this value should be even lower 1e-15
      zz_sparse_log = tf.log(zz_sparse_unlog + 1e-15) 
      zz_sparse = zz_sparse_log
    """

    
    # In log space
    with tf.variable_scope("to_sparse_log") as scope:
      # Fill the sparse matrix with the smallest value in zz (at least -100)
      sparse_filler = tf.minimum(tf.reduce_min(zz), -100)
#       sparse_filler = -100
      zz_sparse = to_sparse(
          zz, 
          spatial_routing_matrix, 
          sparse_filler=sparse_filler)
  
    
    with tf.variable_scope("softmax_across_parents") as scope:
      rr_sparse = softmax_across_parents(zz_sparse, spatial_routing_matrix)
    
    with tf.variable_scope("to_dense") as scope:
      rr_dense = to_dense(rr_sparse, spatial_routing_matrix)
      
    rr = tf.reshape(
        rr_dense, 
        [batch_size, parent_space, parent_space, kh_kw_i, parent_caps, 1])
    #----- End -----#

    # AG 02/11/2018
    # In response to a question on OpenReview, Hinton et al. wrote the 
    # following:
    # "The gradient flows through EM algorithm. We do not use stop gradient. A 
    # routing of 3 is like a 3 layer network where the weights of layers are 
    # shared."
    # https://openreview.net/forum?id=HJWLfGWRb&noteId=S1eo2P1I3Q
    
    return rr


def softmax_across_parents(probs_sparse, spatial_routing_matrix):
  """Softmax across all parent capsules including spatial and depth.

  Consider a sparse matrix of probabilities (1, 5, 5, 49, 8, 32)
  (batch_size, parent_space, parent_space, child_space*child_space, child_caps,   parent_caps)

  For one child capsule, we need to normalise across all parent capsules that
  receive output from that child. This includes the depth of parent capsules,
  and the spacial dimension od parent capsules. In the example matrix of
  probabilities above this would mean normalising across [1, 2, 5] or
  [parent_space, parent_space, parent_caps]. But the softmax function
  `tf.nn.softmax` can only operate across one axis, so we need to reshape the
  matrix such that we can combine paret_space and parent_caps into one axis.

  Author:
    Ashley Gritzman 05/11/2018

  Args:
    probs_sparse:
      the sparse representation of the probs matrix, in log
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)

  Returns:
    rr_updated:
      softmax across all parent capsules, same shape as input
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)
  """

  # e.g. (1, 5, 5, 49, 8, 32)
  # (batch_size, parent_space, parent_space, child_space*child_space,
  # child_caps, parent_caps)
  shape = probs_sparse.get_shape().as_list()
  batch_size = shape[0]
  parent_space = shape[1]
  child_space_2 = shape[3]  # squared
  child_caps = shape[4]
  parent_caps = shape[5]

  # Move parent space dimensions, and parent depth dimension to end
  # (1, 5, 5, 49, 8, 32)  -> (1, 49, 4, 5, 5, 3)
  sparse = tf.transpose(probs_sparse, perm=[0, 3, 4, 1, 2, 5])

  # Combine parent
  # (1, 49, 4, 75)
  sparse = tf.reshape(sparse, [batch_size, child_space_2, child_caps, -1])

  # Perform softmax across parent capsule dimension
  parent_softmax = tf.nn.softmax(sparse, axis=-1)

  # Uncombine parent space and depth
  # (1, 49, 4, 5, 5, 3)
  parent_softmax = tf.reshape(
    parent_softmax,
    [batch_size, child_space_2, child_caps, parent_space, parent_space,
     parent_caps])

  # Return to original order
  # (1, 5, 5, 49, 8, 32)
  parent_softmax = tf.transpose(parent_softmax, perm=[0, 3, 4, 1, 2, 5])

  # Softmax across the parent capsules actually gives us the updated routing
  # weights
  rr_updated = parent_softmax

  # Checks
  # 1. Shape
  assert (rr_updated.get_shape().as_list()
          == [batch_size, parent_space, parent_space, child_space_2,
              child_caps, parent_caps])

  # 2. Check the total of the routing weights is equal to the number of child
  # capsules
  # Note: during convolution some child capsules may be dropped if the
  # convolution doesn't fit nicely. So in the sparse form of child capsules, the   # dropped capsules will be 0 everywhere. When we do a softmax, these capsules
  # will then be given a value, so when we check the total child capsules we
  # need to include these. But these will then be excluded when we convert back   # to dense so it's not a problem.
  total_child_caps = tf.to_float(child_space_2 * child_caps * batch_size)
  sum_routing_weights = tf.round(tf.reduce_sum(rr_updated))

  #   assert_op = tf.assert_equal(
  #       sum_routing_weights,
  #       total_child_caps,
  #       message="""in fn softmax_across_parents: sum_routing_weights and
  #               effective_child_caps are different""")
  #   with tf.control_dependencies([assert_op]):
  #      rr_updated = tf.identity(rr_updated)

  return rr_updated


def init_rr(spatial_routing_matrix, child_caps, parent_caps):
  """Initialise routing weights.

  Initialise routing weights taking into accout spatial position of child
  capsules. Child capsules in the corners only go to one parent capsule, while
  those in the middle can go to kernel*kernel capsules.

  Author:
    Ashley Gritzman 19/10/2018

  Args:
    spatial_routing_matrix:
      A 2D numpy matrix containing mapping between children capsules along the
      rows, and parent capsules along the columns.
      (child_space^2, parent_space^2)
      (7*7, 5*5)
    child_caps: number of child capsules along depth dimension
    parent_caps: number of parent capsules along depth dimension

  Returns:
    rr_initial:
      initial routing weights
      (1, parent_space, parent_space, kk, child_caps, parent_caps)
      (1, 5, 5, 9, 8, 32)
  """

  # Get spatial dimension of parent & child
  parent_space_2 = int(spatial_routing_matrix.shape[1])
  parent_space = int(np.sqrt(parent_space_2))
  child_space_2 = int(spatial_routing_matrix.shape[0])
  child_space = int(np.sqrt(child_space_2))

  # Count the number of parents that each child belongs to
  parents_per_child = np.sum(spatial_routing_matrix, axis=1, keepdims=True)

  # Divide the vote of each child by the number of parents that it belongs to
  # If the striding causes the filter not to fit, it will result in some
  # "dropped" child capsules, which effectively means child capsules that do not
  # have any parents. This would create a divide by 0 scenario, so need to add
  # 1e-9 to prevent NaNs.
  rr_initial = (spatial_routing_matrix
                / (parents_per_child * parent_caps + 1e-9))

  # Convert the sparse matrix to be compatible with votes.
  # This is done by selecting the child capsules belonging to each parent, which
  # is achieved by selecting the non-zero values down each column. Need the
  # combination of two transposes so that order is correct when reshaping
  mask = spatial_routing_matrix.astype(bool)
  rr_initial = rr_initial.T[mask.T]
  rr_initial = np.reshape(rr_initial, [parent_space, parent_space, -1])

  # Copy values across depth dimensions
  # i.e. the number of child_caps and the number of parent_caps
  # (5, 5, 9) -> (5, 5, 9, 8, 32)
  rr_initial = rr_initial[..., np.newaxis, np.newaxis]
  rr_initial = np.tile(rr_initial, [1, 1, 1, child_caps, parent_caps])

  # Add one mode dimension for batch size
  rr_initial = np.expand_dims(rr_initial, 0)

  # Check the total of the routing weights is equal to the number of child
  # capsules
  # child_space * child_space * child_caps (minus the dropped ones)
  dropped_child_caps = np.sum(np.sum(spatial_routing_matrix, axis=1) < 1e-9)
  effective_child_cap = ((child_space * child_space - dropped_child_caps)
                         * child_caps)

  sum_routing_weights = np.sum(rr_initial)

  #   assert_op = tf.assert_less(
  #       np.abs(sum_routing_weights - effective_child_cap), 1e-9)
  #   with tf.control_dependencies([assert_op]):
  #     return rr_initial

  assert np.abs(sum_routing_weights - effective_child_cap) < 1e-3

  return rr_initial


def to_sparse(probs, spatial_routing_matrix, sparse_filler=tf.log(1e-20)):
  """Convert probs tensor to sparse along child_space dimension.

  Consider a probs tensor of shape (64, 6, 6, 3*3, 32, 16).
  (batch_size, parent_space, parent_space, kernel*kernel, child_caps,
  parent_caps)
  The tensor contains the probability of each child capsule belonging to a
  particular parent capsule. We want to be able to sum the total probabilities
  for a single child capsule to all the parent capsules. So we need to convert
  the 3*3 spatial locations have been condensed, into a sparse format across
  all child spatial location e.g. 14*14.

  Since we are working in log space, we must replace the zeros that come about
  during sparse with log(0). The 'sparse_filler' option allows us to specify the
  number to use to fill.

  Author:
    Ashley Gritzman 01/11/2018

  Args:
    probs:
      tensor of log probabilities of each child capsule belonging to a
      particular parent capsule
      (batch_size, parent_space, parent_space, kernel*kernel, child_caps,
      parent_caps)
      (64, 5, 5, 3*3, 32, 16)
    spatial_routing_matrix:
      binary routing map with children as rows and parents as columns
    sparse_filler:
      the number to use to fill in the sparse locations instead of zero

  Returns:
    sparse:
      the sparse representation of the probs tensor in log space
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 7*7, 32, 16)
  """

  # Get shapes of probs
  shape = probs.get_shape().as_list()
  batch_size = shape[0]
  parent_space = shape[1]
  kk = shape[3]
  child_caps = shape[4]
  parent_caps = shape[5]

  # Get spatial dimesion of child capsules
  child_space_2 = int(spatial_routing_matrix.shape[0])
  parent_space_2 = int(spatial_routing_matrix.shape[1])

  # Unroll the probs along the spatial dimension
  # e.g. (64, 6, 6, 3*3, 8, 32) -> (64, 6*6, 3*3, 8, 32)
  probs_unroll = tf.reshape(
    probs,
    [batch_size, parent_space_2, kk, child_caps, parent_caps])

  # Each row contains the children belonging to one parent
  child_to_parent_idx = group_children_by_parent(spatial_routing_matrix)

  # Create an index mapping each capsule to the correct sparse location
  # Each element of the index must contain [batch_position,
  # parent_space_position, child_sparse_position]
  # E.g. [63, 24, 49] maps image 63, parent space 24, sparse position 49
  child_sparse_idx = child_to_parent_idx
  child_sparse_idx = child_sparse_idx[np.newaxis, ...]
  child_sparse_idx = np.tile(child_sparse_idx, [batch_size, 1, 1])

  parent_idx = np.arange(parent_space_2)
  parent_idx = np.reshape(parent_idx, [-1, 1])
  parent_idx = np.repeat(parent_idx, kk)
  parent_idx = np.tile(parent_idx, batch_size)
  parent_idx = np.reshape(parent_idx, [batch_size, parent_space_2, kk])

  batch_idx = np.arange(batch_size)
  batch_idx = np.reshape(batch_idx, [-1, 1])
  batch_idx = np.tile(batch_idx, parent_space_2 * kk)
  batch_idx = np.reshape(batch_idx, [batch_size, parent_space_2, kk])

  # Combine the 3 coordinates
  indices = np.stack((batch_idx, parent_idx, child_sparse_idx), axis=3)
  indices = tf.constant(indices)

  # Convert each spatial location to sparse
  shape = [batch_size, parent_space_2, child_space_2, child_caps, parent_caps]
  sparse = tf.scatter_nd(indices, probs_unroll, shape)

  # scatter_nd pads the output with zeros, but since we are operating
  # in log space, we need to replace 0 with log(0), or log(1e-9)
  zeros_in_log = tf.ones_like(sparse, dtype=tf.float32) * sparse_filler
  sparse = tf.where(tf.equal(sparse, 0.0), zeros_in_log, sparse)

  # Reshape
  # (64, 5*5, 7*7, 8, 32) -> (64, 6, 6, 14*14, 8, 32)
  sparse = tf.reshape(sparse, [batch_size, parent_space, parent_space, child_space_2, child_caps, parent_caps])

  # Checks
  # 1. Shape
  assert sparse.get_shape().as_list() == [batch_size, parent_space, parent_space, child_space_2, child_caps,
                                          parent_caps]

  # This check no longer holds since we have replaced zeros with log(1e-9), so
  # the total of dense and sparse no longer match.
  # 2. Total of dense and sparse must be the same
  #   pct_delta = tf.abs(
  #     (tf.reduce_sum(probs) - tf.reduce_sum(sparse))
  #     /tf.reduce_sum(probs))

  #   assert_op = tf.assert_less(
  #       pct_delta,
  #       1e-4,
  #       message="in fn to_sparse: total of probs and sparse are different",
  #       data=[pct_delta, tf.reduce_sum(probs), tf.reduce_sum(sparse)])
  #   with tf.control_dependencies([assert_op]):
  #      sparse = tf.identity(sparse)

  return sparse


def to_dense(sparse, spatial_routing_matrix):
  """Convert sparse back to dense along child_space dimension.

  Consider a sparse probs tensor of shape (64, 5, 5, 49, 8, 32).
  (batch_size, parent_space, parent_space, child_space*child_space, child_caps,
  parent_caps)
  The tensor contains all child capsules at every parent spatial location, but
  if the child does not route to the parent then it is just zero at that spot.
  Now we want to get back to the dense representation:
  (64, 5, 5, 49, 8, 32) -> (64, 5, 5, 9, 8, 32)

  Author:
    Ashley Gritzman 05/11/2018
  Args:
    sparse:
      the sparse representation of the probs tensor
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)
    spatial_routing_matrix:
      binary routing map with children as rows and parents as columns

  Returns:
    dense:
      the dense representation of the probs tensor
      (batch_size, parent_space, parent_space, kk, child_caps, parent_caps)
      (64, 5, 5, 9, 8, 32)
  """

  # Get shapes of probs
  shape = sparse.get_shape().as_list()
  batch_size = shape[0]
  parent_space = shape[1]
  child_space_2 = shape[3]  # squared
  child_caps = shape[4]
  parent_caps = shape[5]

  # Calculate kernel size by adding up column of spatial routing matrix
  kk = int(np.sum(spatial_routing_matrix[:, 0]))

  # Unroll parent spatial dimensions
  # (64, 5, 5, 49, 8, 32) -> (64, 5*5, 49, 8, 32)
  sparse_unroll = tf.reshape(sparse, [batch_size, parent_space * parent_space,
                                      child_space_2, child_caps, parent_caps])

  # Apply boolean_mask on axis 1 and 2
  # sparse_unroll: (64, 5*5, 49, 8, 32)
  # spatial_routing_matrix: (49, 25) -> (25, 49)
  # dense: (64, 5*5, 49, 8, 32) -> (64, 5*5*9, 8, 32)
  dense = tf.boolean_mask(sparse_unroll,
                          tf.transpose(spatial_routing_matrix), axis=1)

  # Reshape
  dense = tf.reshape(dense, [batch_size, parent_space, parent_space, kk,
                             child_caps, parent_caps])

  # Checks
  # 1. Shape
  assert (dense.get_shape().as_list()
          == [batch_size, parent_space, parent_space, kk, child_caps,
              parent_caps])

  #   # 2. Total of dense and sparse must be the same
  #   delta = tf.abs(tf.reduce_sum(dense, axis=[3])
  #                  - tf.reduce_sum(sparse, axis=[3]))
  #   assert_op = tf.assert_less(
  #       delta,
  #       1e-6,
  #       message="in fn to_dense: total of dense and sparse are different",
  #       data=[tf.reduce_sum(dense,[1,2,3,4,5]),
  #             tf.reduce_sum(sparse,[1,2,3,4,5]),
  #             tf.reduce_sum(dense),tf.reduce_sum(sparse)],
  #       summarize=10)
  #   with tf.control_dependencies([assert_op]):
  #      dense = tf.identity(dense)

  return dense


def group_children_by_parent(bin_routing_map):
  """Groups children capsules by parent capsule.

  Rearrange the bin_routing_map so that each row represents one parent capsule,   and the entries in the row are indexes of the children capsules that route to   that parent capsule. This mapping is only along the spatial dimension, each
  child capsule along in spatial dimension will actually contain many capsules,   e.g. 32. The grouping that we are doing here tell us about the spatial
  routing, e.g. if the lower layer is 7x7 in spatial dimension, with a kernel of
  3 and stride of 1, then the higher layer will be 5x5 in the spatial dimension.
  So this function will tell us which children from the 7x7=49 lower capsules
  map to each of the 5x5=25 higher capsules. One child capsule can be in several
  different parent capsules, children in the corners will only belong to one
  parent, but children towards the center will belong to several with a maximum   of kernel*kernel (e.g. 9), but also depending on the stride.

  Author:
    Ashley Gritzman 19/10/2018
  Args:
    bin_routing_map:
      binary routing map with children as rows and parents as columns
  Returns:
    children_per_parents:
      parents are rows, and the indexes in the row are which children belong to       that parent
  """

  tmp = np.where(np.transpose(bin_routing_map))
  children_per_parent = np.reshape(tmp[1], [bin_routing_map.shape[1], -1])

  return children_per_parent
