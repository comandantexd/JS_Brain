class Neuron {
    // array of inputs, array of
    constructor() {
        this.inputs = [];
        this.weigths = [];
        this.net = 0;
        this.out = 0;
        this.bias = 1;
    }

    input(input) {
        this.inputs = input;

        for (let i = 0; i < this.inputs.length; i++) {
            this.weigths[i] = Math.random();
        }
    }

    process() {
        //calculating the weighted average
        this.net = 0;
        for (let i = 0; i < this.inputs.length; i++) {
            this.net += this.inputs[i] * this.weigths[i];
        }
        this.net += this.bias;

        //Passing the sigmoid activation function to the weighted average
        this.out = 1/(1 + Math.pow(Math.E, -this.net));
        return this.out;
    }
}

let lay = [2,1];
let ins = [1,1];

class Network {
    constructor(layer) {
        this.ERROR_MARGIN = 0.001;

        this.netOut = 0; //final output of the network
        this.netErr = 0; //total error of the network

        //initialization of layers and neurons and the outputs for each layer
        this.layer = new Array(layer.length);
        this.out = new Array(layer.length);
        for (let i = 0; i < layer.length; i++) { //layers
            this.layer[i] = new Array(layer[i]);
            this.out[i] = new Array(layer[i]);
            for (let c = 0; c < layer[i]; c++) { //neurons
                this.layer[i][c] = new Neuron();
            }
        }
        console.log(this.layer);
    }

    process(inputs, target) {
        this.target = target;
        this.inputs = inputs;

        for (let i = 0; i < this.layer.length; i++) { //layers
            for (let c = 0; c < this.layer[i].length; c++) { //neurons

                if (i == 0) {
                    this.layer[i][c].input(this.inputs)
                    this.out[i][c] = this.layer[i][c].process();
                    console.log(`Input for layer ${i} y neurona ${c}: ${this.inputs}`)
                } else {
                    this.layer[i][c].input(this.out[i - 1]);
                    this.out[i][c] = this.layer[i][c].process();
                    console.log(`Input for layer ${i} y neurona ${c}: ${this.out[i - 1]}`)

                    if (i == this.layer.length - 1) { //checks if is last layer
                        this.netOut = this.out[i][c];
                    }
                }
            }
        }
        console.log(this.out);

        //checking the error
        this.netErr = (1/2) * Math.pow(this.target - this.netOut, 2);
        if (this.netErr > this.ERROR_MARGIN) {
            console.log(`error de la red: ${this.netErr}`);
            this.backpropagation();
        }
    }

    // this.gradient = -(this.target - this.out) * this.out * (1 - this.out) * this.inputs[i];
    // this.weigths[i] -= 0.5 * this.gradient;

    // TODO: Finish backpropagation algorithm--
    backpropagation() {
        for (let i = this.layer.length - 1; i >= 0; i--) { //layers from back to front
            for (let c = 0; c < this.layer[i].length; c++) { //neurons
                if (i == this.layer.length - 1) {
                    //ultima capa

                    for (let w = 0; w < this.layer[i][c].weigths.length; w++) { //por alguna razon el error no baja
                        this.gradient = -(this.target - this.netOut) * this.netOut * (1 - this.netOut) * this.out[i - 1][w];

                        this.layer[i][c].weigths[w] -= 0.5 * this.gradient; // actualizando el peso
                    }
                } else if (i == 0) {
                    //primera capa, el input de esta es el input original


                } else {
                    //resto de capas, el input de estas es el output de la anterior

                }
            }
        }
    }
}




//for debug
let net = new Network(lay);
console.clear();
net.process(ins, 0);
