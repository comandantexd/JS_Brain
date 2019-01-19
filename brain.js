class Neuron {
    constructor(input1, input2, target) {
        this.guess = false;

        this.i1 = input1;
        this.i2 = input2;
        this.target = target;

        this._g_err_o;
        this._g_o_pondered;

        this._w1 = Math.random();
        this._g_pondered_w1;
        this._g_w1;

        this._w2 = Math.random();
        this._g_pondered_w2;
        this._g_w2;

        this._b = 1;

        this._pondered;
        this._o;
        this._err;
    }

    input(input1, input2, target) {
        this.i1 = input1;
        this.i2 = input2;
        this.target = target;
    }

    dendrites() { //ponderes the inputs with the weights
        this._pondered = (this.i1 * this._w1) + (this.i2 * this._w2) + this._b;
        this.soma();
        return;
    }

    soma() { //applyes the activation function of sigmoid
        this._o = 1/(1 + (Math.pow(Math.E, -this._pondered)));
        this.axon();
        return;
    }

    axon(){ //prints the output and calculates the total error
        this._err = 1/2 * Math.pow(this.target - this._o, 2);
        if (this.guess) {
            console.log(this._o);
        } else {
            if (this._err > 0.1) {
                this.backpropagation();
                return;
            } else {
                console.log(this._o);
            }
        }
    }

    backpropagation() {
        this._g_err_o = -(this.target - this._o);
        this._g_o_pondered = this._o * (1 - this._o);
        this._g_pondered_w1 = this.i1;
        this._g_pondered_w2 = this.i2;

        this._g_w1 = this._g_err_o * this._g_o_pondered * this._g_pondered_w1;
        this._g_w2 = this._g_err_o * this._g_o_pondered * this._g_pondered_w2;

        this._w1 = this._w1 - 0.5 * this._g_w1;
        this._w2 = this._w2 - 0.5 * this._g_w2;
        this.dendrites();
        return;
    }
}
