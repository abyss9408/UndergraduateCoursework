/**
 * This is the main entry point for the application. 
 * - initializes the BabylonJS engine
 * - creates the scene and runs the render loop
 *
 * If you edit this file, you MUST add your authorship 
 * information below, including your declaration of 
 * originality (see README.md).
 * Failure to declare authorship may be treated as a 
 * violation of academic integrity.
 */

import { Engine } from '@babylonjs/core';
import { App } from './app';

// get the canvas element
const canvas = <HTMLCanvasElement>document.getElementById('renderCanvas');

// initialize babylon engine
const engine = new Engine(canvas, true);

// create the scene and run the render loop
const app = new App(engine);
app.createScene().then(scene => {
    engine.runRenderLoop(() => {
        scene.render();
    })
});

// resize the canvas when the window is resized
window.addEventListener('resize', function () {
    engine.resize();
});
