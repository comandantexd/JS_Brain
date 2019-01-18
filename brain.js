//una unica neurona con 2 entradas

var v1;
var v2;

var w1 = 1;
var w2 = 1;
var b = 0;

const selection = 0.5;

var target = true;
var activated_r;

function sinapsis(a, b) {
    let s1 = a * w1;
    let s2 = b * w2;

    ponderacion(s1,s2);
}

function ponderacion(s1, s2) {
    v1 = s1;
    v2 = s2;

    let f = s1 + s2 + b;

    activacion(f);
}

function activacion(y) {
    y = 1 / (1 + (Math.pow(Math.E, -y)));

    result(y);
}

function result(y) {
    let final;
    activated_r = y;

    activated_r >= selection ? final = true : final = false;

    if (target == "na") {
        console.log('Guessed result: ' + final);
    } else {
        final == target ? console.log(final) : backpropagation(activated_r);
    }
}

function backpropagation(a) {
    if (target) { //needs to ve false
        b += activated_r;
        sinapsis(v1, v2);
    } else { //needs to be true
        b -= activated_r;
        sinapsis(v1, v2);
    }
    console.log('fixed');
}