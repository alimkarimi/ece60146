from ComputationalGraphPrimer import *

class Subclasses_ADAM(ComputationalGraphPrimer):
    def __init__(self):
        print("in subclass!!")
        super(Subclasses_ADAM, self).__init__(   num_layers = 3,
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
     ) #initialize base class ComputationalGraphPrimer 
        self.loss_oneN
        self.loss_multiN
    
    def somefunc(self, x):
        print(x)

    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid, mt_list, vt_list, mt_bias_list, vt_bias_list, iteration):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        B1 = 0.9
        B2 = 0.99
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            #print("len(mt_list), len(vt_list):", len(mt_list), len(vt_list))
            ## Calculate the next step in the parameter hyperplane
#            step = self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid    
            #step = self.learning_rate * y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid #this is Gt+1
            grad = y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid
            if ((len(mt_list) <4 ) and (len(vt_list) < 4)):
                mt_list.append((1 - B1) * grad)
                vt_list.append((1 - B2) * grad**2)
                mt_list[i] = mt_list[i] / (1 - B1) #technically, B1 is raised to the power of the iteration, but in this case iter = 1  
                vt_list[i] = vt_list[i] / (1 - B2) #technically, B2 is raised to the power of the iteration, but in this case iter = 1
            else:
                power = iteration + 1
                #print("raising to the power of", power)
                mt_list.append(B1 * mt_list[-4] + ((1 - B1) * grad))
                vt_list.append(B2 * vt_list[-4] + ((1 - B2) * grad**2))
                mt_list[-1] = mt_list[-1] / (1 - B1**power)
                vt_list[-1] = vt_list[-1] / (1 - B2**power)
            ## Update the learnable parameters
            epsilon = 1e-08
            self.vals_for_learnable_params[param] = self.vals_for_learnable_params[param] + ((self.learning_rate * (mt_list[-1]) / (np.sqrt(vt_list[-1] + epsilon))))
            #self.vals_for_learnable_params[param] += step #original step update
        bias_gradient = y_error * deriv_sigmoid
        #print("len(mt_bias_list), len(vt_bias_list): " ,len(mt_bias_list), len(vt_bias_list))
        #print("length of mt_bias_list", len(mt_bias_list) )
        if ((len(mt_bias_list) == 0) and (len(vt_bias_list) == 0)):
            mt_bias_list.append((1 - B1) * bias_gradient)
            mt_bias_list[0] = mt_bias_list[0] / (1 - B1) #technically, B1 is raised to power of iteration, but, here, B1^1, so it is not explicitly written
            vt_bias_list.append((1 - B2) * bias_gradient**2)
            vt_bias_list[0] = vt_bias_list[0] / (1 - B2)
        else:
            power = iteration + 1
            #print("power is", power)
            #print("raising to the power of", power)
            mt_bias_list.append(B1 * mt_bias_list[-1] + (1 - B1) * bias_gradient)
            vt_bias_list.append(B2 * vt_bias_list[-1] + (1 - B2) * bias_gradient**2)
            mt_bias_list[-1] = mt_bias_list[-1] / (1 - B1**power)
            vt_bias_list[-1] = vt_bias_list[-1] / (1 - B2**power)
        self.bias = self.bias + (self.learning_rate * (mt_bias_list[-1] / np.sqrt(vt_bias_list[-1] + epsilon)))
        #self.bias += self.learning_rate * y_error * deriv_sigmoid    ## Update the bias

    def backprop_and_update_params_multi_neuron_model(self, y_error, class_labels, mt_list, vt_list, mt_bias_list, vt_bias_list, iteration):
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
        B1 = 0.9
        B2 = 0.99
        epsilon = 1e-08
        pred_err_backproped_at_layers = {i : [] for i in range(1,self.num_layers-1)}  
        pred_err_backproped_at_layers[self.num_layers-1] = [y_error]
        for back_layer_index in reversed(range(1,self.num_layers)):
            input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]
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
                ##  Regarding the parameter update loop that follows, see the Slides 74 through 77 of my Week 3 
                ##  lecture slides for how the parameters are updated using the partial derivatives stored away 
                ##  during forward propagation of data. The theory underlying these calculations is presented 
                ##  in Slides 68 through 71. 
                for i,param in enumerate(layer_params):
                    gradient_of_loss_for_param = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] * deriv_sigmoid_avg[j]
                    #step = self.learning_rate * gradient_of_loss_for_param * deriv_sigmoid_avg[j]
                    power = iteration + 1
                    if (len(mt_list) < 10 and len(vt_list) < 10):
                        #initialize vt and mt to 0
                        mt_list.append((1 - B1) * gradient_of_loss_for_param) 
                        vt_list.append((1 - B2) * gradient_of_loss_for_param**2)
                        mt_list[-1] = mt_list[-1] / (1 - B1)
                        vt_list[-1] = vt_list[-1] / (1 - B2)
                    else:
                        mt_list.append(B1 * mt_list[-10] + (1 - B1) * gradient_of_loss_for_param)
                        vt_list.append(B2 * vt_list[-10] + (1 - B2) * gradient_of_loss_for_param**2)
                        mt_list[-1] = mt_list[-1] / (1 - B1**power)
                        vt_list[-1] = vt_list[-1] / (1 - B2**power) 
                    self.vals_for_learnable_params[param] += self.learning_rate * mt_list[-1] / np.sqrt(vt_list[-1] + epsilon)
            power = iteration + 1
            bias_gradient = sum(pred_err_backproped_at_layers[back_layer_index]) \
                                                                           * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg)        
            if ((len(mt_bias_list) < 2) and (len(vt_bias_list) < 2)):
                mt_bias_list.append((1 - B1) * bias_gradient)
                vt_bias_list.append((1 - B2) * bias_gradient**2)
                mt_bias_list[-1] = mt_bias_list[-1] / (1-B1)
                vt_bias_list[-1] = vt_bias_list[-1] / (1 - B2)
            else: 
                mt_bias_list.append((B1 * mt_bias_list[-2]) + (1 - B1) * bias_gradient)
                vt_bias_list.append((B2 * vt_bias_list[-2]) + (1 - B2) * bias_gradient**2)
                mt_bias_list[-1] = mt_bias_list[-1] / (1 - B1**power)
                vt_bias_list[-1] = vt_bias_list[-1] / (1 - B2**power)
                #print("power is", power)
            #print('k is')
            #print('mt_bias:', mt_bias_list[-1])
            #print('vt_bias:', vt_bias_list[-1])    
            self.bias[back_layer_index-1] = self.bias[back_layer_index-1] + self.learning_rate * (mt_bias_list[-1] / np.sqrt(vt_bias_list[-1] + epsilon))
            # self.bias[back_layer_index-1] += self.learning_rate * sum(pred_err_backproped_at_layers[back_layer_index]) \
            #                                                                * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg)
    
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
                mt_list = []
                vt_list = []
                mt_bias_list = []
                vt_bias_list = []
            iteration = i
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
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))            ## Display avg loss
                avg_loss_over_iterations = 0.0                                                ## Re-initialize avg-over-iterations loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            self.backprop_and_update_params_multi_neuron_model(y_error_avg, class_labels, mt_list, vt_list, mt_bias_list, vt_bias_list, iteration)      ## BACKPROP loss
        plt.figure()     
        plt.plot(loss_running_record) 
        self.loss_multiN = loss_running_record
        #plt.show()   
    
    def run_training_loop_one_neuron_model(self, training_data):
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
                mt_list = []
                vt_list = []
                vt_bias_list = []
                mt_bias_list = []
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
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg, mt_list, vt_list, mt_bias_list, vt_bias_list, iteration)     ## BACKPROP loss
        plt.figure()     
        plt.plot(loss_running_record)
        self.loss_oneN = loss_running_record 
        #plt.show()   

# adam = Subclasses_ADAM()
# adam.somefunc(5)
# adam.parse_expressions()
# adam.display_one_neuron_network()      
# training_data = adam.gen_training_data()
# adam.run_training_loop_one_neuron_model( training_data )

# adam = Subclasses_ADAM()
# adam.somefunc(5)
# adam.parse_multi_layer_expressions()
# adam.display_multi_neuron_network()      
# training_data = adam.gen_training_data()
# adam.run_training_loop_multi_neuron_model( training_data )