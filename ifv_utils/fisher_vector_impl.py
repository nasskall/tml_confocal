import numpy as np
from numpy.linalg import matrix_power


def fisher_vector_mel(xx, gmm):
    xx = np.atleast_2d(xx)  # de kanei tpt apla sigoureyei oti einai 2d an den einai to forsarei
    N = xx.shape[0]  # arithos descriptors

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK gyrnaei pinaka me tis pithanothtes toy kathe descriptor na anoikei
    # sti kathe gausiani

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:,
            np.newaxis] / N  # np.sum(Q, 0) athrisma stilwn, [:, np.newaxis] metatropi grammis se stili kai ola /N kai einai oustiastika o mean toy Q
    Q_xx = np.dot(Q.T,
                  xx) / N  # eswteriko ginomeno toy Q me xx afou exoyme bgalei anastrofo toy Q gia na ginei h praksi
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N  # eswteriko ginomeno tou Q me xx^2
    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_  # afairese ta barh twn gausianwn apo ton mean tou Q ( average ψ(πk) )
    d_mu = np.linalg.inv(gmm.covariances_) * (
            Q_xx - Q_sum * gmm.means_)  # ψ(μk) alla to original to xei xwris to /gmm.covariances!!
    d_sigma = (  # ψ(σk) alla me diaforetika proshma kai apo to original leipei to /sqrt(2) * gmm.covariances ** 2
                      Q_xx_2
                      + Q_sum * gmm.means_ ** 2
                      - Q_sum * matrix_power(gmm.covariances_, 2)
                      - 2 * Q_xx * gmm.means_) * 1 / np.sqrt(2) * matrix_power(np.linalg.inv(gmm.covariances_), 2)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

def fisher_vector(xx, gmm):
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.covariances_
            + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))