class Neuron {
    // array of inputs, array of
    constructor() {
        this.inputs = [];
        this.weigths = [];
        this.net = 0;
        this.out = 0;
        this.err = 0;
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

//Calculating the error of the neuron
//this.err = (1/2) * Math.pow(this.target - this.out, 2);

let lay = [2,2,1];
let ins = [1,1];
let tar = 1;

class Network {
    constructor(inputs, layer) {
        this.netOut = 0;

        //initialization of layers and neurons
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
        for (let i = 0; i < this.layer.length; i++) { //layers
            for (let c = 0; c < this.layer[i].length; c++) { //neurons

                if (i == 0) {
                    this.layer[i][c].input(inputs)
                    this.out[i][c] = this.layer[i][c].process();
                    console.log(`Input for layer ${i} y neurona ${c}: ${inputs}`)
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
    }
    // backpropagation() {
    //     for (let i = 0; i < this.inputs.length; i++) {
    //         this.gradient = -(this.target - this.out) * this.out * (1 - this.out) * this.inputs[i];
    //         this.weigths[i] -= 0.5 * this.gradient;
    //     }
    // }
}
