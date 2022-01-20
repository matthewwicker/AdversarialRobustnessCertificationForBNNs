# Author: Matthew Wicker

# Advisory indices

# COC=0
# WL=1
# WR=2
# SL=3
# SR=4


def phi_0(iml, imu, ol, ou, numOut=5):
    v1 = tf.one_hot(TRUE_VALUE, depth=numOut)
    v2 = 1 - tf.one_hot(TRUE_VALUE, depth=numOut)
    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
    worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
    if(np.argmax(worst_case) == TRUE_VALUE):
        return True
    else:
        return False

def phi_0(iml, imu, ol, ou, numOut=5):
    v1 = tf.one_hot(TRUE_VALUE, depth=numOut)
    v2 = 1 - tf.one_hot(TRUE_VALUE, depth=numOut)
    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
    worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
    if(np.argmax(worst_case) == TRUE_VALUE):
        return True
    else:
        return False
