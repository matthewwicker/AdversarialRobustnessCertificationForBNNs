import numpy as np

# !! THIS IS ONLY FOR RELU !! NO OTHER ACTIVATION SUPPORTED atm
def get_alphas_betas(zeta_l, zeta_u, activation="relu"):
    alpha_L, alpha_U = list([]), list([])
    beta_L, beta_U = list([]), list([])
    for i in range(len(zeta_l)):
        if(zeta_u[i] <= 0):
            alpha_U.append(0); alpha_L.append(0); beta_L.append(0); beta_U.append(0)
        elif(zeta_l[i] >= 0):
            alpha_U.append(1); alpha_L.append(1); beta_L.append(0); beta_U.append(0)
        else:
            # For relu I have the points (zeta_l, 0) and (zeta_u, zeta_u)
            a_U = zeta_u[i]/(zeta_u[i]-zeta_l[i]); b_U = -1*(a_U*zeta_l[i])
    
            #a_L = a_U ; b_L = 0
            #if (zeta_u[i] + zeta_l[i]) >= 0:
            #    a_L = 1 ;   b_L = 0
            #else:
            a_L = 0 ;   b_L = 0    
            alpha_U.append(a_U); alpha_L.append(a_L); beta_L.append(b_L); beta_U.append(b_U)
    return alpha_U, beta_U, alpha_L, beta_L



def get_bar_lower(linear_bound_coef, mu_l, mu_u,
                  nu_l, nu_u, lam_l, lam_u):
    mu_l = np.squeeze(mu_l); mu_u = np.squeeze(mu_u); 
    mu_bar, nu_bar, lam_bar = [], [], []
    
    nu_bar = nu_l

    #coef of the form - alpha_U, beta_U, alpha_L, beta_L
    for i in range(len(linear_bound_coef)):
        if(linear_bound_coef[i,2] >= 0):
            mu_bar.append(linear_bound_coef[i,2] * mu_l[i])
            for k in range(len(nu_bar)):
                try:
                    nu_bar[k][i] = linear_bound_coef[i,2] * np.asarray(nu_l[k][i])
                except:
                    print 'error'
            lam_bar.append(linear_bound_coef[i,2] * lam_l[i] + linear_bound_coef[i,3])
        else:
            mu_bar.append(linear_bound_coef[i,2] * mu_u[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i,2] * nu_u[k][i]
            lam_bar.append(linear_bound_coef[i,2] * lam_u[i] + linear_bound_coef[i,3])
    return np.asarray(mu_bar), nu_bar, np.asarray(lam_bar)

def get_bar_upper(linear_bound_coef, mu_l, mu_u,
                  nu_l, nu_u, lam_l, lam_u):
    mu_l = np.squeeze(mu_l); mu_u = np.squeeze(mu_u);  
    mu_bar, nu_bar, lam_bar = [], [], []
    nu_bar = nu_u
    for i in range(len(linear_bound_coef)):
        if(linear_bound_coef[i,0] >= 0):
            mu_bar.append(linear_bound_coef[i,0] * mu_u[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i,0] * np.asarray(nu_u[k][i])
            lam_bar.append(linear_bound_coef[i,0] * lam_u[i] + linear_bound_coef[i,1])
        else:
            mu_bar.append(linear_bound_coef[i,0] * mu_l[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i,0] * nu_l[k][i]
            lam_bar.append(linear_bound_coef[i,0] * lam_l[i] + linear_bound_coef[i,1])
    return np.asarray(mu_bar), nu_bar, np.asarray(lam_bar)

def get_abc_lower(w, mu_l_bar, nu_l_bar, la_l_bar,
               mu_u_bar, nu_u_bar, la_u_bar):
    a, b, c = [], [], []
    for i in range(len(w)):
        curr_a = []
        #curr_b = []
        curr_c = []
        for j in range(len(w[i])):
            if(w[i][j] >= 0):
                curr_a.append(w[i][j] * mu_l_bar[i])
                curr_c.append(w[i][j] * la_l_bar[i])
            else:
                curr_a.append(w[i][j] * mu_u_bar[i])
                curr_c.append(w[i][j] * la_u_bar[i])
        a.append(curr_a)
        
        c.append(curr_c)
    for k in range(len(nu_l_bar)): 
        curr_b = []
        #for i in range(len(w)):
        for j in range(len(w[i])):
            curr_curr_b = []
            #for j in range(len(w[i])):
            for i in range(len(w)):
                if(w[i][j] >= 0):
                    curr_curr_b.append(w[i][j] * nu_l_bar[k][i])
                else:
                    curr_curr_b.append(w[i][j] * nu_u_bar[k][i])
            curr_b.append(curr_curr_b)
        b.append(curr_b)  
        
        
    return np.asarray(a), b, np.asarray(c)


def get_abc_upper(w, mu_l_bar, nu_l_bar, la_l_bar,
               mu_u_bar, nu_u_bar, la_u_bar):
    #This is anarchy
    return get_abc_lower(w,mu_u_bar, nu_u_bar, la_u_bar,
                         mu_l_bar, nu_l_bar, la_l_bar)


def min_of_linear_fun(coef_vec, uppers, lowers):
   #getting the minimum
    val_min = 0
    for i in range(len(coef_vec)):
        if coef_vec[i] >=0:
            val_min = val_min + coef_vec[i]*lowers[i]
        else: 
            val_min = val_min + coef_vec[i]*uppers[i]
    return val_min

def max_of_linear_fun(coef_vec, uppers, lowers):
    val_max = - min_of_linear_fun(-coef_vec, uppers, lowers)
    return val_max


def propogate_lines(x, in_reg, sWs,sbs,
                    w_margin=0.25, search_samps=100, act = 'relu'):

    x = np.asarray(x); x = x.astype('float64')
    x_l, x_u = in_reg[0], in_reg[1]
    try:
        loaded_model = np.load(model_path, allow_pickle=True)
        [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
        dWs = [dW_0,dW_1]
        dbs = [db_0,db_1]
        widths = [512]   
    except:
        with open(model_path, 'r') as pickle_file:
            [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = pickle.load(pickle_file)
            dWs = [dW_0,dW_1]
            dbs = [db_0,db_1]
            widths = [512]   
   
      
    n_hidden_layers = len(widths)
    
    #Code adaptation end. From now on it's the standard code 
     
    #Step 1: Inputn layers -> Pre-activation function        
    W_0_L, W_0_U, b_0_L, b_0_U = (sWs[0][0] - dWs[0]*w_margin,  sWs[0][0] + dWs[0]*w_margin, 
                                  sbs[0][0]-dbs[0]*w_margin, sbs[0][0]+dbs[0]*w_margin)
    
    W_0_L = W_0_L.T
    W_0_U = W_0_U.T
    
    mu_0_L = W_0_L; mu_0_U = W_0_U
    
    n_hidden_1 = sWs[0][0].shape[1]
    
    nu_0_L = np.asarray([x_l for i in range(n_hidden_1) ])
    nu_0_U = np.asarray([x_l for i in range(n_hidden_1) ])
    la_0_L = - np.dot(x_l, W_0_L.T) + b_0_L
    la_0_U = - np.dot(x_l, W_0_U.T) + b_0_U
    
    
    # getting bounds on pre-activation fucntion
    zeta_0_L = [ (min_of_linear_fun(np.concatenate((mu_0_L[i].flatten(), nu_0_L[i].flatten())), 
                                     np.concatenate((np.asarray(x_u).flatten(), W_0_U[i].flatten() )),
                                     np.concatenate((np.asarray(x_l).flatten(), W_0_L[i].flatten() ))  )) for i in range(n_hidden_1)] 
   
    zeta_0_L = np.asarray(zeta_0_L) + la_0_L
     
    zeta_0_U = [ (max_of_linear_fun(np.concatenate((mu_0_U[i].flatten(), nu_0_U[i].flatten())),
                                     np.concatenate((np.asarray(x_u).flatten(), W_0_U[i].flatten())),
                                     np.concatenate((np.asarray(x_l).flatten(), W_0_L[i].flatten()))  )) for i in range(n_hidden_1)]
        
    zeta_0_U = np.asarray(zeta_0_U) + la_0_U
    
    
    #Initialising variable for main loop
    curr_zeta_L = zeta_0_L
    curr_zeta_U = zeta_0_U
    curr_mu_L = mu_0_L
    curr_mu_U = mu_0_U
    curr_nu_L = [nu_0_L]
    curr_nu_U = [nu_0_U]
    curr_la_L = la_0_L
    curr_la_U = la_0_U
    
    W_Ls = W_0_L.flatten()
    W_Us = W_0_U.flatten()
    #loop over the hidden layers
    for l in range(1,n_hidden_layers+1):
        if l < n_hidden_layers:
            curr_n_hidden = widths[l]
        else:
            curr_n_hidden = 1
            
        LUB = np.asarray(get_alphas_betas(curr_zeta_L, curr_zeta_U))
        LUB = np.asmatrix(LUB).transpose() 
        # Now evaluate eq (*) conditions:
        curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar = get_bar_lower(LUB, curr_mu_L, curr_mu_U, 
                                                           curr_nu_L, curr_nu_U, 
                                                           curr_la_L, curr_la_U)

        curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar = get_bar_upper(LUB, curr_mu_L, curr_mu_U, 
                                                           curr_nu_L, curr_nu_U, 
                                                           curr_la_L, curr_la_U)
        
        curr_z_L = [   min_of_linear_fun( [LUB[i,2]] , [curr_zeta_U[i]] , [curr_zeta_L[i]]     ) + LUB[i,3]
                      for i in range(len(curr_zeta_U))    ]

        #SUpper and lower bounds for weights and biases of current hidden layer
        curr_W_L, curr_W_U, curr_b_L, curr_b_U = (sWs[l][0] - dWs[l]*w_margin,  sWs[l][0] + dWs[l]*w_margin,
                                      sbs[l][0] - dbs[l]*w_margin, sbs[l][0] + dbs[l]*w_margin)
    
        a_L, b_L, c_L = get_abc_lower(curr_W_L, curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar,
                               curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar)
        
        a_U, b_U, c_U = get_abc_upper(curr_W_U, curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar,
                               curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar)
        
        curr_mu_L = np.sum(a_L, axis=0)
        curr_mu_U = np.sum(a_U, axis=0)
        curr_nu_L = []
        curr_nu_U = []
        for k in range(l-1):
            curr_nu_L.append(np.sum(b_L[k], axis=1))
            curr_nu_U.append(np.sum(b_U[k], axis=1))
        
        curr_nu_L.append(b_L[l-1])
        curr_nu_U.append(b_U[l-1])
        
        
        
        
        curr_nu_L.append(np.asarray([curr_z_L for i in range(curr_n_hidden) ]))
        curr_nu_U.append(np.asarray([curr_z_L for i in range(curr_n_hidden) ]))
        
            
        curr_la_L = np.sum(c_L, axis=0) - np.dot(curr_z_L, curr_W_L) + curr_b_L
        curr_la_U = np.sum(c_U, axis=0) - np.dot(curr_z_L, curr_W_U) + curr_b_U
    

            
        curr_zeta_L = []
        curr_zeta_U = []
        
        for i in range(curr_n_hidden):
            ith_mu_L = curr_mu_L[i]
            ith_mu_U = curr_mu_U[i]

            
            ith_W_Ls = np.concatenate( (W_Ls, curr_W_L.T[i]) )
            ith_W_Us = np.concatenate( (W_Us, curr_W_U.T[i]) )
            ith_nu_L = []
            ith_nu_U = []
            for k in range(len(curr_nu_L)):
                ith_nu_L = np.concatenate(  ( ith_nu_L, np.asarray(curr_nu_L[k][i]).flatten()  )    )
                ith_nu_U = np.concatenate(  ( ith_nu_U, np.asarray(curr_nu_U[k][i]).flatten()  )    )
                
               
            curr_zeta_L.append( min_of_linear_fun( np.concatenate( (ith_mu_L, ith_nu_L) ) ,
                                                       np.concatenate( (x_u, ith_W_Us     ) ) ,
                                                       np.concatenate( (x_l, ith_W_Ls     ) )
                                                      )   )  
            
            curr_zeta_U.append( max_of_linear_fun( np.concatenate( (ith_mu_U, ith_nu_U) ) ,
                                                   np.concatenate( (x_u, ith_W_Us     ) ) ,
                                                   np.concatenate( (x_l, ith_W_Ls     ) )
                                                  )   ) 
        curr_zeta_L  = curr_zeta_L + curr_la_L
        curr_zeta_U  = curr_zeta_U + curr_la_U
        
        W_Ls = np.concatenate((W_Ls ,   curr_W_L.T.flatten()  ))
        W_Us = np.concatenate((W_Us ,   curr_W_U.T.flatten()  ))
        
    #Code adaptation for output:    
    #end code adaptation for output
    return [curr_zeta_L, curr_zeta_U]