class Neuron {
    // array of inputs, array of
    constructor(inputs) {
        this.inputs = [];
        this.weigths = [];
        this.net = 0;
        this.out = 0;
        this.bias = -2; //can be modified

        for (let i = 0; i < inputs; i++) {
            this.weigths[i] = Math.random();
        }
    }

    input(input) {
        this.inputs = input;
    }

    process() {
        //calculating the weighted average
        this.net = 0;
        for (let i = 0; i < this.inputs.length; i++) {
            this.net = this.net + (this.inputs[i] * this.weigths[i]);
        }
        this.net += this.bias;

        //Passing the sigmoid activation function to the weighted average
        this.out = 1 / (1 + (Math.pow(Math.E, 0 - this.net)));

        return this.out;
    }
}
//-------------------------------------------------------------------------------------------------------------------------

class Network {
    constructor(layer, inputs) { // layer as ana rray, inputs as number of inputs
        this.ERROR_MARGIN = 0.001;
        this.LEARN_RATE = 0.3;
        this.guest = false;

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
        //console.log(this.layer);
    }

    process(inputs, target) {
        this.target = target;
        this.inputs = inputs;

        for (let i = 0; i < this.layer.length; i++) { //layers
            for (let c = 0; c < this.layer[i].length; c++) { //neurons
                if (i == 0) {
                    this.layer[i][c].input(this.inputs)
                    this.out[i][c] = this.layer[i][c].process();
                    //console.log(`Input for layer ${i} y neurona ${c}: ${this.inputs}`)
                } else {
                    this.layer[i][c].input(this.out[i - 1]);
                    this.out[i][c] = this.layer[i][c].process();
                    //console.log(`Input for layer ${i} y neurona ${c}: ${this.out[i - 1]}`)

                    if (i == this.layer.length - 1) { //checks if is last layer
                        this.netOut = this.out[i][c];
                    }
                }
            }
        }

        this.netErr = (1/2) * Math.pow(this.target - this.netOut, 2); // COST FUNCTION
        //console.log(this.netOut);
        if (this.netErr > this.ERROR_MARGIN && this.netErr != NaN && this.guest == false) {
            //console.log(`error de la red: ${this.netErr}`);
            this.backpropagation();
            return;
        } else {
            console.log(this.netOut);
            this.memory();
        }
    }

    // TODO: Finish backpropagation algorithm--
    backpropagation() {
        let gradient = 0;
        let neuronErr = 0;
        let weightMatrix = 0;
        for (let i = this.layer.length - 1; i >= 0; i--) { //layers from back to front
            for (let c = 0; c < this.layer[i].length; c++) { //neurons
                if (i == this.layer.length - 1) {
                    //ultima capa

                    for (let w = 0; w < this.layer[i][c].weigths.length; w++) { //por cada peso de la neurona
                        neuronErr = (this.netOut - this.target) * (this.netOut * (1 - this.netOut));
                        gradient = neuronErr * this.out[i-1][w];
                        //console.log(`modificanco del peso ${w} de la neurona ${c} de la capa ${i}: ${this.out[i-1][w]}`);
                        this.layer[i][c].weigths[w] -= this.LEARN_RATE * gradient;
                    }

                } else if (i == 0) {
                    //primera capa, el input de esta es el input original

                    for (let w = 0; w < this.layer[i][c].weigths.length; w++) {
                        weightMatrix += this.layer[i][c].weigths[w];
                    }

                    neuronErr = neuronErr * weightMatrix * (this.out[i][c] * (1 - this.out[i][c]));

                    for (let w = 0; w < this.layer[i][c].weigths.length; w++)  {
                        gradient = neuronErr * this.inputs[w];
                        this.layer[i][c].weigths[w] -= this.LEARN_RATE * gradient;
                    }

                } else {
                    //resto de capas, el input de estas es el output de la anterior

                    for (let w = 0; w < this.layer[i][c].weigths.length; w++) {
                        weightMatrix += this.layer[i][c].weigths[w];
                    }

                    neuronErr = neuronErr * weightMatrix * (this.out[i][c] * (1 - this.out[i][c]));

                    //console.log(this.out, i - 1);
                    for (let w = 0; w < this.layer[i][c].weigths.length; w++)  {
                        gradient = neuronErr * this.out[i-1][w];
                        this.layer[i][c].weigths[w] -= this.LEARN_RATE * gradient;
                    }
                }
            }
        }
        this.process(this.inputs, this.target);
        return;
    }
}

let lay = [3,1]; //[nodes_L0, nodes_L1]
let net = new Network(lay, 20);
