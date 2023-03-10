from ComputationalGraphPrimer import *
from CGP_subclasses_ADAM import *

loss_in_run = []
loss_sgd_plus = []
loss_adam = []
loss_sgd_plus_MN = []
class Subclasses_SGD_plus(ComputationalGraphPrimer):
    def __init__(self):
        print("in subclass!!")
        super(Subclasses_SGD_plus, self).__init__(  num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               #learning_rate = 1e-6,
               learning_rate = 1e-3,
               #learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
     ) #initialize base class ComputationalGraphPrimer 
    
    def somefunc(self, x):
        print(x)

    def run_training_loop_one_neuron_model(self, training_data): 
        #### DERIVED VERSION OF ONE NEURON TRAINING LOOP
        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                          ##   We initialize it to a random number.

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how 
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)
           
            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of 
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                           ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data
                batch = [batch_data, batch_labels]
                return batch                


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                           ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            if (i == 0):
                step_list = [] #add in list to remember step sizes. 
                print(step_list)
                bias_vt1_list = []
            iteration = i

            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)              ##  FORWARD PROP of data
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ##  Find loss
            loss_avg = loss / float(len(class_labels))                                              ##  Average the loss over batch
            avg_loss_over_iterations += loss_avg                          
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                loss_in_run.append(avg_loss_over_iterations) #global variable -- for plotting in HW assignment
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg, step_list, bias_vt1_list, iteration)     ## BACKPROP loss
        plt.figure()     
        plt.plot(loss_running_record) 
        #plt.show()   

    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid, vt1_list, bias_vt1_list, k):
        """
        As should be evident from the syntax used in the following call to backprop function,
        self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.
        
        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        #print("IN THE DERIVED VERSION OF BACKPROP!!")
        momentum = 0.99
        #print(type(deriv_sigmoid))
        #print("this is deriv_sigmoid", deriv_sigmoid)
        input_vars = self.independent_vars
        #print("the input_vars:", input_vars)
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]
        #print("the input_vars_to_param_map:", input_vars_to_param_map)
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}
        #print(param_to_vars_map)
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        #print(vals_for_input_vars_dict)
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            #print("len of vt_list is", len(vt1_list))
            #print("in iteration", i, "and the param is", param)
            ## Calculate the next step in the parameter hyperplane
            #            step = self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid    
            #print("learning rate is", self.learning_rate, "y_error is", y_error, "deriv_sigmoid is", deriv_sigmoid, ". With this computing step")
           
            gradient =   y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid  #orig step computation. removed self.learning_rate and put a 2 in
             ####THIS CODE BELOW INITIALIZES Vt TO 0 IF THERE IS NO PREV ITERATION. 
             ### IF THERE IS NO PREV ITER: the step size is just equal to the gradient Gt+1, which was computed above as step 
             ### IF THERE IS A PREV ITERATION. Vt+1 = MOMENTUM * Vt + GRADIENT (computed as the variable "step" in code already)
            if (len(vt1_list) <4 ):
                mu_vt = 0 #momentum * vt = momentum * 0 = 0
                vt1_list.append(mu_vt + gradient) #append Vt+1 to step list. We will need to access this as Vt for future computations.
            else:
                vt1 = (momentum * vt1_list[-4]) + gradient
                vt1_list.append(vt1)
            #step = (momentum * prev_step) + step #here, step is Vt+1, momentum is mu, step_list[i-1] is Vt, and step is Gt+1
            #print("step size for param", param, "is", step)
            ## Update the learnable parameters
            #self.vals_for_learnable_params[param] += gradient #from orig code: equivalent to self.vals_for_learnable_params[param] = self.vals_for_learnable_params[param] + step
            self.vals_for_learnable_params[param] = self.vals_for_learnable_params[param] + (self.learning_rate * vt1_list[-1]) #implement -momentum * dpk-1
        #self.bias += self.learning_rate * y_error * deriv_sigmoid    ## Update the bias
        bias_gradient = y_error * deriv_sigmoid
        if (len(bias_vt1_list) == 0):
            mu_vt = 0
            bias_vt1_list.append(mu_vt + bias_gradient)
        else:
            bias_vt1 = (momentum * bias_vt1_list[-1]) + bias_gradient
            bias_vt1_list.append(bias_vt1)
        self.bias = self.bias + self.learning_rate * bias_vt1_list[-1] #(self.learning_rate * y_error * deriv_sigmoid)
        #print("self.bias is", self.bias)

    ######################################################################################################



    def backprop_and_update_params_multi_neuron_model(self, y_error, class_labels, Vt1_list, bias_vt1_list):
        """
        First note that loop index variable 'back_layer_index' starts with the index of
        the last layer.  For the 3-layer example shown for 'forward', back_layer_index
        starts with a value of 2, its next value is 1, and that's it.

        Stochastic Gradient Gradient calls for the backpropagated loss to be averaged over
        the samples in a batch.  To explain how this averaging is carried out by the
        backprop function, consider the last node on the example shown in the forward()
        function above.  Standing at the node, we look at the 'input' values stored in the
        variable "input_vals".  Assuming a batch size of 8, this will be list of
        lists. Each of the inner lists will have two values for the two nodes in the
        hidden layer. And there will be 8 of these for the 8 elements of the batch.  We average
        these values 'input vals' and store those in the variable "input_vals_avg".  Next we
        must carry out the same batch-based averaging for the partial derivatives stored in the
        variable "deriv_sigmoid".

        Pay attention to the variable 'vars_in_layer'.  These store the node variables in
        the current layer during backpropagation.  Since back_layer_index starts with a
        value of 2, the variable 'vars_in_layer' will have just the single node for the
        example shown for forward(). With respect to what is stored in vars_in_layer', the
        variables stored in 'input_vars_to_layer' correspond to the input layer with
        respect to the current layer. 
        """
        # backproped prediction error:
        #print("In multi neuron model!!!")
        momentum = 0.99
        pred_err_backproped_at_layers = {i : [] for i in range(1,self.num_layers-1)}  
        #print("pred_error_backproped at layers:" ,pred_err_backproped_at_layers)
        pred_err_backproped_at_layers[self.num_layers-1] = [y_error]
        #print("pred_error_backproped at layers:" ,pred_err_backproped_at_layers)
        #print("this is y_error:", y_error)

        for back_layer_index in reversed(range(1,self.num_layers)):
            #print("\n\n\nSTARTING NEW LOOP IN BACKLAYER INDEX\n\n\n")
            #print("this is back_layer_index:", back_layer_index)
            input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]
            #print("this is input_vals", input_vals)
            input_vals_avg = [sum(x) for x in zip(*input_vals)]
            input_vals_avg = list(map(operator.truediv, input_vals_avg, [float(len(class_labels))] * len(class_labels)))
            deriv_sigmoid =  self.gradient_vals_for_layers[back_layer_index]
            deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
            deriv_sigmoid_avg = list(map(operator.truediv, deriv_sigmoid_avg, 
                                                             [float(len(class_labels))] * len(class_labels)))
            vars_in_layer  =  self.layer_vars[back_layer_index]                 ## a list like ['xo']
            vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]   ## a list like ['xw', 'xz']

            layer_params = self.layer_params[back_layer_index]         
            ## note that layer_params are stored in a dict like        
                ##     {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
            ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
            transposed_layer_params = list(zip(*layer_params))         ## creating a transpose of the link matrix

            backproped_error = [None] * len(vars_in_next_layer_back)
            for k,varr in enumerate(vars_in_next_layer_back):
                for j,var2 in enumerate(vars_in_layer):
                    backproped_error[k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]] * 
                                               pred_err_backproped_at_layers[back_layer_index][i] 
                                               for i in range(len(vars_in_layer))])
#                                               deriv_sigmoid_avg[i] for i in range(len(vars_in_layer))])
            pred_err_backproped_at_layers[back_layer_index - 1]  =  backproped_error
            input_vars_to_layer = self.layer_vars[back_layer_index-1]


                
            for j,var in enumerate(vars_in_layer):
                layer_params = self.layer_params[back_layer_index][j]
                #print("j is", j)
                #print("layer params are:", layer_params)
                ##  Regarding the parameter update loop that follows, see the Slides 74 through 77 of my Week 3 
                ##  lecture slides for how the parameters are updated using the partial derivatives stored away 
                ##  during forward propagation of data. The theory underlying these calculations is presented 
                ##  in Slides 68 through 71. 
                for i,param in enumerate(layer_params):
                    #print("len(step_list):", len(Vt1_list))
                    #print("currently in param", param)
                        
                    #print("i is", i)
                    gradient_of_loss_for_param = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] * deriv_sigmoid_avg[j]
                    #print("in iteration", i, "gradient_of_loss_for_param is", gradient_of_loss_for_param )
                    #step = self.learning_rate * gradient_of_loss_for_param * deriv_sigmoid_avg[j]
                    if len(Vt1_list) < 10: #for the first 10 parameters, Vt = 0, so, Vt+1 = momentum * Vt + Gt+1 will be just be Gt+1
                        #which was computed in step
                        mu_vt = 0
                        #print("step is:" ,step)
                        Vt1_list.append(mu_vt + gradient_of_loss_for_param)
                    else:
                        Vt1 = momentum * Vt1_list[-10] + gradient_of_loss_for_param
                        Vt1_list.append(Vt1)
                    self.vals_for_learnable_params[param] += self.learning_rate * Vt1_list[-1] #from original code. Here, step size was computed as dependent on 
                    #just the gradient.
                    #self.vals_for_learnable_params[param] = self.vals_for_learnable_params[param] - prev_step #update for SGD+
            bias_gradient = sum(pred_err_backproped_at_layers[back_layer_index]) \
                                                                           * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg)
            if (len(bias_vt1_list) <2 ):
                mu_vt = 0
                bias_vt1_list.append(mu_vt + bias_gradient)
            else:
                bias_vt1_list.append(momentum * bias_vt1_list[-2] + bias_gradient)
            #print("len(bias_vt1_list):", len(bias_vt1_list))
            self.bias[back_layer_index-1] += self.learning_rate * bias_vt1_list[-1]        
            #self.bias[back_layer_index-1] += self.learning_rate * sum(pred_err_backproped_at_layers[back_layer_index]) \
            #                                                               * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg)
            #print(step_list)
    ######################################################################################################


    def run_training_loop_multi_neuron_model(self, training_data):

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how 
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)
           
            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of 
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]    ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]    ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                           ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data       
                batch = [batch_data, batch_labels]
                return batch                


        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = [random.uniform(0,1) for _ in range(self.num_layers-1)]      ## Adding the bias to each layer improves 
                                                                                 ##   class discrimination. We initialize it 
                                                                                 ##   to a random number.

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                          ##  Average the loss over iterations for printing out 
                                                                                 ##    every N iterations during the training loop.   
        for i in range(self.training_iterations):
            if (i == 0):
                step_list = [] #add in list to remember step sizes.
                bias_vt1_list = [] 
                #print(step_list)
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                  ## FORW PROP works by side-effect 
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]      ## Predictions from FORW PROP
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]  ## Get numeric vals for predictions
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ## Calculate loss for batch
            loss_avg = loss / float(len(class_labels))                                         ## Average the loss over batch
            avg_loss_over_iterations += loss_avg                                              ## Add to Average loss over iterations
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                loss_in_run.append(avg_loss_over_iterations) #global variable -- for plotting in HW assignment
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))            ## Display avg loss
                avg_loss_over_iterations = 0.0                                                ## Re-initialize avg-over-iterations loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            self.backprop_and_update_params_multi_neuron_model(y_error_avg, class_labels, step_list, bias_vt1_list)      ## BACKPROP loss
        plt.figure()     
        plt.plot(loss_running_record) 
        #plt.show()   


# # #CODE FOR SGD+ ONE NEURON NTWK
# sub = Subclasses_SGD_plus()
# sub.parse_expressions()
# #sub.display_one_neuron_network()      
# training_data = sub.gen_training_data()
# sub.run_training_loop_one_neuron_model( training_data )
# loss_sgd_plus = loss_in_run #copy current loss into another obj so that we can store losses for normal SGD run in loss
# # ###END CODE FOR SGD+ ONE NEURON NTWK

# # ###CODE FOR SGD ONE NEURON NETWORK #####
# cgp = ComputationalGraphPrimer(
#                one_neuron_model = True,
#                expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
#                output_vars = ['xw'],
#                dataset_size = 5000,
#                learning_rate = 1e-3,
#                #learning_rate = 5 * 1e-2,
#                training_iterations = 40000, #originally 40000
#                batch_size = 8,
#                display_loss_how_often = 100, #orig 100
#                debug = True,
#       )
# cgp.parse_expressions()
# #cgp.display_network1()
# #cgp.display_network2()
# #cgp.display_one_neuron_network()      
# training_data = cgp.gen_training_data()
# cgp.run_training_loop_one_neuron_model( training_data )
# loss_sgd_orig = cgp.loss_oneN
# print(cgp.loss_oneN)
# # # ### END CODE FOR SGD ONE NEURON NETWORK #####

# ###CODE FOR ADAM ONE NEURON ####
# adam = Subclasses_ADAM()
# adam.somefunc(5)
# adam.parse_expressions()
# #adam.display_one_neuron_network()      
# training_data = adam.gen_training_data()
# adam.run_training_loop_one_neuron_model( training_data )
# loss_adam = adam.loss_oneN
# ### END CODE FOR ADAM ONE NEURON ###


# ###NEED TO UPDATE PLOTTING LOGIC. RIGHT NOW, NOTHING IS WRITTEN INTO loss_sgd_orig because I haven't figured out how 
# # to get that list from the function that is running it (in the base class). The over-ridden function in this file, but the 
# # function running the original SGD is in the base class. 
# x_axis_len = len(loss_sgd_plus)
# # x_axis_len = len(loss_adam)
# #print(loss_sgd_plus)
# #print('sgd_orig:', loss_sgd_orig)
# plt.plot(np.linspace(0, x_axis_len, x_axis_len), loss_sgd_plus, label = "SGD+")
# plt.plot(np.linspace(0, x_axis_len, x_axis_len), loss_sgd_orig, label = "SGD")
# plt.plot(np.linspace(0, x_axis_len, x_axis_len), loss_adam, label = "ADAM")
# plt.xlabel("Training Iter")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("One Neuron Plots - LR =" + str(adam.learning_rate))
# plt.show()

#CODE FOR SGD+ MULTI NEURON NETWORK
sub = Subclasses_SGD_plus()
sub.parse_multi_layer_expressions()
#sub.display_multi_neuron_network()   
training_data = sub.gen_training_data()
sub.run_training_loop_multi_neuron_model( training_data )
loss_sgd_plus_MN = loss_in_run
##END CODE FOR SGD+ MULTI NEURON NETWORK


##CODE FOR SGD MULTI NEURON NETWORK #####
cgp = ComputationalGraphPrimer(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               #learning_rate = 1e-6,
               #learning_rate = 1e-3,
               learning_rate = 5 * 1e-2,
               #learning_rate = 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )
cgp.parse_multi_layer_expressions()
#cgp.display_multi_neuron_network()   
training_data = cgp.gen_training_data()
cgp.run_training_loop_multi_neuron_model( training_data )
loss_sgd_orig = cgp.loss_multiN
## END CODE FOR SGD MULTI NEURON NETWORK #####



##CODE FOR ADAM MULTI NEURON ####
adam = Subclasses_ADAM()
adam.parse_multi_layer_expressions()
#adam.display_multi_neuron_network
training_data = adam.gen_training_data()
adam.run_training_loop_multi_neuron_model(training_data)
loss_adam_multiN = adam.loss_multiN
##END CODE FOR ADAM MULTI NEURON



x_axis_len = len(loss_adam_multiN)
plt.plot(np.linspace(0, x_axis_len, x_axis_len), loss_sgd_plus_MN, label = "SGD+")
plt.plot(np.linspace(0, x_axis_len, x_axis_len), loss_sgd_orig, label = "SGD")
plt.plot(np.linspace(0, x_axis_len, x_axis_len), loss_adam_multiN, label = "ADAM")
plt.xlabel("Training Iter")
plt.ylabel("Loss")
plt.legend()
plt.title("Multi Neuron Plot w/ LR=" + str(adam.learning_rate))
plt.show()