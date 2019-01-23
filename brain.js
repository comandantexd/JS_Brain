class Neuron {
    // array of inputs, array of
    constructor(inputs) { //numeric
        this.inputs = [];
        this.weights = [];
        this.net = 0;
        this.out = 0;
        this.bias = Math.random() * (1 - (-1)) + (-1);

        for (let i = 0; i < inputs; i++) {
            this.weights[i] = Math.random() * (1 - (-1)) + (-1);
        }
    }

    process(inputs) { //array
        this.inputs = inputs;

        //calculating the weighted average
        this.net = 0;
        for (let i = 0; i < this.inputs.length; i++) {
            this.net = this.net + (this.inputs[i] * this.weights[i]);
        }
        this.net += this.bias;

        //Passing the sigmoid activation function to the weighted average
        this.out = 1 / (1 + (Math.pow(Math.E, -this.net)));

        return this.out;
    }
}

//-------------------------------------------------------------------------------------------------------------------------

class Network {
    constructor(layer, inputs) { // layer as an array, inputs as number of inputs
        this.ERROR_MARGIN = 0.0001;
        this.LEARN_RATE = 0.1;
        this.ITERATIONS = 10000;

        this.netOut = 0; //final output of the network
        this.netErr = 0; //total error of the network

        //initialization of layers and neurons and the outputs for each layer
        this.layer = new Array(layer.length);
        this.out = new Array(layer.length);

        for (let i = 0; i < layer.length; i++) { //layers
            this.layer[i] = new Array(layer[i]);
            this.out[i] = new Array(layer[i]);

            for (let c = 0; c < layer[i]; c++) { //neurons
                this.layer[i][c] = new Neuron(i == 0 ? inputs : layer[i-1]);
            }
        }
    }

    process(inputs) {
        this.inputs = inputs;

        for (let i = 0; i < this.layer.length; i++) { //layers
            for (let c = 0; c < this.layer[i].length; c++) { //neurons
                if (i == 0) {
                    this.out[i][c] = this.layer[i][c].process(this.inputs);

                } else {
                    this.out[i][c] = this.layer[i][c].process(this.out[i - 1]);

                    if (i == this.layer.length - 1) { //checks if is last layer
                        this.netOut = this.out[i][c];
                    }
                }
            }
        }
        return this.netOut;
    }

    train(inputs,target) {
        let ins = 0;
        let error = new Array();
        let _w;
        let _err;
        do {
            this.inputs = inputs[ins % inputs.length];
            this.target = target[ins % inputs.length];
            ins++;

            for (let i = 0; i < this.layer.length; i++) { //layers
                for (let c = 0; c < this.layer[i].length; c++) { //neurons
                    if (i == 0) {
                        this.out[i][c] = this.layer[i][c].process(this.inputs);

                    } else {
                        this.out[i][c] = this.layer[i][c].process(this.out[i - 1]);

                        if (i == this.layer.length - 1) { //checks if is last layer
                            this.netOut = this.out[i][c];
                        }
                    }
                }
            }

            this.netErr = (1/2) * Math.pow(this.target - this.netOut, 2); // COST FUNCTION

            //BACKPROPAGATION FUNCTION
            for (let i = 0; i < this.layer.length; i++) { //layers
                error[i] = new Array(this.layer[i].length);
            }

            //first calculate the error for each neuron
            for (let i = this.layer.length - 1; i >= 0; i--) {
                for (let c = 0; c < this.layer[i].length; c++) {
                    if (i == this.layer.length - 1) {

                        error[i][c] = (this.netOut - this.target) * this.netOut * (1 - this.netOut);

                    } else {
                        _w = 1;
                        _err = 0;
                        error[i][c] = 1;

                        for (let n = 0; n < this.layer[i+1].length; n++) {
                            for (let w = 0; w < this.layer[i+1][n].weights.length; w++){
                                _err += error[i+1][n] * this.layer[i+1][n].weights[w];
                            }
                        }

                        error[i][c] = _err * this.out[i][c] * (1 - this.out[i][i]);
                    }
                }
            }

            //update the weights and bias according with the error of each neuron
            for (let i = 0; i < this.layer.length; i++) {
                for (let c = 0; c < this.layer[i].length; c++) {
                    for (let w = 0; w < this.layer[i][c].weights.length; w++) {
                        this.layer[i][c].weights[w] -= this.LEARN_RATE * error[i][c] *  this.layer[i][c].inputs[w];
                    }

                    this.layer[i][c].bias -= this.LEARN_RATE * error[i][c];
                }
            }

        } while (this.netErr > this.ERROR_MARGIN && typeof(target) != "undefined" && ins < this.ITERATIONS);

        if (ins >= this.ITERATIONS) {
            return 0;
        }
    }
}

// JUST FOR EXAMPLE
// let lay = [3,6,1]; //[nodes_L0, nodes_L1]
// let net = new Network(lay, 2);
//
// let train = [
//     [1,1],
//     [1,0],
//     [0,1],
//     [0,0]
// ];
// let targets = [0,1,1,0];
//
// net.train(train, targets);
//net.process(arr_of_inputs); for asking data
