import numpy as np 


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) *(180/np.pi)


def main():
    dim_baseline = 512
    count_basis = 60000
    list_of_basis = []
    save_basis_address = "./attribute_model/random_reset_train.npy"

    for i in range(count_basis):
        sample1 = np.random.randn(dim_baseline)
        sample1 = sample1 / np.linalg.norm(sample1)
        sample1 = np.sqrt(dim_baseline) * sample1
        list_of_basis.append(sample1)
        # list_of_basis.append(sample1/np.linalg.norm(sample1))
    
    # for i in range(len(list_of_basis)):
    #     for j in range(i+1,len(list_of_basis)):
    #         sample1 = list_of_basis[i]
    #         sample2 = list_of_basis[j]
    #         angle = angle_between(sample1, sample2)
    #         print ("angle", angle)
    #         assert (angle > 80 and angle < 96) 


        print("===================")
    np.save(save_basis_address,list_of_basis)    


if __name__ == "__main__":
    main()
