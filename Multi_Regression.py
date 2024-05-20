import numpy as np
import scipy.stats as st
import warnings


class RegressioninputCheck:
    @staticmethod
    def array_batch_check(array_list):
        if type(array_list) is np.ndarray:
            multiple_num_list = np.array(
                [1] * 1 if len(array_list.shape) <= 2 else array_list.shape[0],
                dtype=int)
            return array_list, multiple_num_list
        else:
            # Judgement of sub list in the whole list
            has_list = any(isinstance(sub_item, list) and len(
                np.array(sub_item).shape) > 1 for sub_item in array_list)

            if has_list:
                sub_list_length = [len(sub_list_item)
                                   for sub_list_item in array_list]
                # Getting the least common multiple
                least_common_multiple = np.lcm.reduce(sub_list_length)
                multiple_num_list = [least_common_multiple] * \
                    len(sub_list_length) / np.array(sub_list_length, dtype=int)
                array_list = [
                    array_list[index] * int(multiple_num_list[index]) for index in range(len(sub_list_length))]
            else:
                multiple_num_list = [int(1)]

            return np.array(array_list, dtype=np.float32), multiple_num_list

    def __init__(self, strain_variable, independent_variable, batch_check):
        self.strain_variable = strain_variable
        self.independent_variable = independent_variable

        self.batch_check = batch_check

        if type(self.strain_variable) is np.ndarray and type(self.independent_variable) is np.ndarray:
            self.batch_check = False
            # if推导表达式
            self.multiple_num_list = np.array(
                [1] * 1 if len(independent_variable.shape) <= 2 else independent_variable.shape[0], dtype=int)

        if self.batch_check:
            warnings.warn("""Warining! This method may cooupy huge memory space, which may lead 'Out of Memory' error. 
      Therefore, please process youre data in advance when executing the regression method (use numpy array), and
      setting `batch_check=False`.""", UserWarning)
            self.strain_variable, _ = self.array_bacth_check(
                self.strain_variable)
            self.independent_variable, self.multiple_num_list = self.array_batch_check(
                self.independent_variable)

        self.dim_i_v = self.independent_variable.shape
        self.dim_s_v = self.strain_variable.shape

        self.dim_i_v_length = len(self.dim_i_v)
        self.dim_s_v_length = len(self.dim_s_v)

    def check_logic(self):
        if (self.dim_s_v_length > 3 or self.dim_i_v_length > 3) or (self.dim_s_v_length == 0 or self.dim_i_v_length == 0):
            print("Sorry, this method does not support dimension over 3 or less than 1, please check your input \n"
                  "strain variable has {} dimensions, and independent variable has {} dimensions".format(self.dim_s_v_length, self.dim_i_v_length))

            return None, None

        if self.dim_s_v_length > 1 and self.dim_s_v[-1] != 1:
            print("The last axis of strain varaible must be 1, you got {}".format(
                self.dim_s_v[-1]))

            return None, None

        max_dim = np.max((self.dim_s_v_length, self.dim_i_v_length))

        try:
            if self.dim_s_v_length < max_dim:
                self.strain_variable = self.strain_variable.reshape(
                    self.dim_s_v[: -1].__add__((1,)))

            if self.dim_i_v_length < max_dim:
                self.independent_variable = self.independent_variable.reshape(self.dim_s_v[:, -1].__add__(
                    (np.int(np.cumprod(self.dim_i_v)[-1] / np.cumprod(self.dim_s_v[: -1])[-1]),)))

            self.dim_s_v_length = len(self.strain_variable.shape)
            self.dim_i_v_length = len(self.strain_variable.shape)

        except Exception as e:
            print("Sorry, your strain varaible and independent variable did not match, the error message is listed as follow: \n"
                  "{}".format(e))

            return None, None

        while self.dim_s_v_length < 3 or self.dim_i_v_length < 3:
            self.strain_variable = np.expand_dims(self.strain_variable, axis=0)
            self.independent_variable = np.expand_dims(
                self.independent_variable, axis=0)
            self.dim_s_v_length = len(self.strain_variable.shape)
            self.dim_i_v_length = len(self.strain_variable.shape)

        return self.strain_variable, self.independent_variable


def regression(strain_variable, independent_variable, with_constant=False, batch_check=False, confidence_level=0.95):
    data_object = RegressioninputCheck(
        strain_variable, independent_variable, batch_check)
    multiple_num_list = np.expand_dims(data_object.multiple_num_list, axis=-1)
    strain_variable, independent_variable = data_object.check_logic()

    if strain_variable is None or independent_variable is None:
        print("Regression init failed, check your input.")
        return None

    if with_constant:
        independent_variable = np.c_[independent_variable,
                                     np.array([1] * np.cumprod(independent_variable.shape[: -1])[-1]).reshape(independent_variable.shape[: -1].__add__((1,)))]
        i_v = independent_variable

    # Setting degree of gifreedom
        df_regression = i_v.shape[2] - 1
        df_residual = i_v.shape[1] / multiple_num_list - df_regression - 1
        df_total = i_v.shape[1] / multiple_num_list - 1
    else:
        i_v = independent_variable
        df_regression = i_v.shape[2]
        df_residual = i_v.shape[1] / multiple_num_list - df_regression
        df_total = i_v.shape[1] / multiple_num_list

    i_v_t = independent_variable.transpose(0, 2, 1)
    s_v = strain_variable

    # Getting regression coefficient
    coeff = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(i_v_t, i_v)), i_v_t), s_v)

    # Getting the residuals
    epsilon = s_v - np.matmul(i_v, coeff)
    epsilon_t = epsilon.transpose(0, 2, 1)

    # Getting residual covariance matrix
    RCM = np.matmul(epsilon_t, epsilon) / epsilon.shape[1]

    # Calculating the variance
    # Note: The original formula will use np.kron() and np.diagonal method to calculate the final answer.
    # At this time, we just use np.multiply() and np.einsum(), which adjust to the batch computing
    # Original Code:
    # `var = len(epsilon) / (len(epsilon) -len(temp.columns)) * np.diagonal(
    #         np.kron(RCM, np.linalg.inv(np.matmul(i_v_t, i_v))))`
    # More than that, the `np.matmul(i_v_t, i_v)` and `len(epsilon) / len(epsilon) - len(temp.columns)` should be
    # scaled.（等比例缩放）
    var = np.einsum('...ii->...i',
                    RCM * np.linalg.inv(np.matmul(i_v_t, i_v) /
                                        np.expand_dims(multiple_num_list, axis=-1))
                    ) * ((epsilon.shape[1] / multiple_num_list) / ((i_v.shape[1] / multiple_num_list) - i_v.shape[2]))

    # Calculating the standard error
    std_error = var ** 0.5

    # Calculating the t-statistics
    t_stat = coeff / np.expand_dims(std_error, axis=-1)

    # Getting total sum of squares
    SST = np.sum(np.square(s_v - np.expand_dims(np.mean(s_v,
                 axis=1), axis=-1)), axis=1) / multiple_num_list

    # Getting sum of squared error
    SSE = np.sum(np.square(epsilon), axis=1) / multiple_num_list

    # Getting sum of squares of the regression
    SSR = SST - SSE

    # Getting r-square
    R_2 = SSR / SST

    # Getting Adjusted r-square
    A_R_2 = 1 - (1 - R_2) * df_total / df_residual

    # Getting mean square of error
    MSE = SSE / (i_v.shape[1] / multiple_num_list)

    # Getting mean square error of the regression
    MS_Regression = SSR / df_regression

    # Getting mean square error of the residuals
    MS_Residuals = SSE / df_residual

    # Getting the F-value
    F = MS_Regression / MS_Residuals

    p_value_F = st.f.sf(F, df_regression, df_residual)

    # p-value should use two-tailed T-test
    p_value_T = st.t.sf(
        np.abs(t_stat), np.expand_dims(df_residual, axis=-1)) * 2

    return {
        "beta": coeff,
        "standard_error_of_coefficients": std_error,
        "t_test_statistic_of_coefficients": t_stat,
        "probability_of_t_test": p_value_T,
        "R_square": R_2,
        "Adjusted_R_square": A_R_2,
        "sum_squares_of_resudials": SSE,
        "sum_squares_of_regression": SSR,
        "total_sum_of_squares": SST,
        "mean_square_error_of_regression": MS_Regression,
        "mean_square_error_of_residuals": MS_Residuals,
        "F_value": F,
        "probability_of_f_test": p_value_F,
        "mean_square_of_error": MSE,
    }


if __name__ == '__main__':
    a = np.array([
        [0.045, 0.7675, 0.9894],
        [0.6974, 0.8037, 0.483],
        [0.0122, 0.2731, 0.2712],
        [0.1066, 0.8387, 0.7594],
        [0.895, 0.4189, 0.1135],
        [0.7045, 0.9625, 0.9052],
        [0.3319, 0.0443, 0.0924],
        [0.6851, 0.8677, 0.96],
        [0.972, 0.0268, 0.1102],
        [0.76, 0.9853, 0.5848],
    ])

    b = np.array([
        [3.2798],
        [14.7512],
        [0.9826],
        [3.7064],
        [18.0641],
        [15.8782],
        [6.9142],
        [15.8283],
        [19.7701],
        [15.9833],
    ])

    result = regression(b, a, with_constant=True, batch_check=True)
    print(result)