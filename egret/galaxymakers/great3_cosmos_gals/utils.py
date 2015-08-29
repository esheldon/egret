import numpy as np
import galsim

def great3_sn_comp(stamp,galinfo):
    # G08 is the best possible S/N estimate:
    #   S = sum W(x,y) I(x,y) / sum W(x,y)
    #   N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
    # with W(x,y) = I(x,y), so
    #   S = sum I^2(x,y) / sum I(x,y)
    #   N^2 = noise variance * sum I^2(x,y) / (sum I(x,y))^2
    #   S/N = sqrt(sum I^2(x,y)) / sqrt(noise variance)
    actual_sn_g08 = np.sqrt((stamp.array**2).sum() / float(galinfo['noise_builder_params']['variance']))
    try:
        res = stamp.FindAdaptiveMom()
        aperture_noise = np.sqrt(float(galinfo['noise_builder_params']['variance']) * \
                                    2.*np.pi*(res.moments_sigma**2))
        # The number below is the flux S/N within an elliptical Gaussian filter.  My
        # guess is that it will be somewhere below the optimal actual_sn_g08 but not too
        # horrible.
        sn_ellip_gauss = res.moments_amp / aperture_noise
        # We also want to estimate the S/N on the size, using an unweighted estimator
        #   S = Sum I(x,y) [(x-x_c)^2 + (y-y_c)^2]
        #   N^2 = (noise variance) * Sum [(x-x_c)^2 + (y-y_c)^2]^2
        # For this, we use the centroid estimate from the adaptive moments.  But we also
        # have to set up the grid of x, y values for the postage stamp, according to the
        # same exact convention as used for adaptive moments, which is that the center
        # of the first pixel is 1.  I do not like this estimator because if we make the
        # postage stamp larger (with white space) then S doesn't change but N^2
        # changes.  So let's instead use a weighted version:
        #   S = Sum W(x,y) I(x,y) [(x-x_c)^2 + (y-y_c)^2] / Sum W(x,y)
        #   N^2 = (noise variance) * Sum W^2(x,y) [(x-x_c)^2 + (y-y_c)^2]^2 /
        #                                      (Sum W(x,y))^2
        # Use W(x,y) = I(x,y),
        #   S = Sum I(x,y)^2 [(x-x_c)^2 + (y-y_c)^2] / Sum I(x,y)
        #   N^2 = (noise variance) * Sum I^2(x,y) [(x-x_c)^2 + (y-y_c)^2]^2 /
        #                                      (Sum I(x,y))^2
        #   S/N = Sum I(x,y)^2 [(x-x_c)^2 + (y-y_c)^2] /
        #         sqrt[(noise variance) * Sum I^2(x,y) [(x-x_c)^2 + (y-y_c)^2]^2]
        if stamp.array.shape[0] != stamp.array.shape[1]:
            raise RuntimeError
        min = 1.
        max = float(stamp.array.shape[0]+1)
        x_pix, y_pix = np.meshgrid(np.arange(min, max, 1.),
                                      np.arange(min, max, 1.))
        dx_pix = x_pix - (res.moments_centroid.x - (res.image_bounds.xmin-1))
        dy_pix = y_pix - (res.moments_centroid.y - (res.image_bounds.ymin-1))
        sn_size = np.sum(stamp.array**2 * (dx_pix**2 + dy_pix**2)) / \
                  np.sqrt(float(galinfo['noise_builder_params']['variance']) * \
                             np.sum(stamp.array**2 * (dx_pix**2 + dy_pix**2)**2))
    except:
        sn_ellip_gauss = -10.
        sn_size = -10.

    sndict = {}
    sndict['sn_G08'] = actual_sn_g08
    sndict['sn_ellip_gauss'] = sn_ellip_gauss
    sndict['sn_size'] = sn_size
    return sndict
