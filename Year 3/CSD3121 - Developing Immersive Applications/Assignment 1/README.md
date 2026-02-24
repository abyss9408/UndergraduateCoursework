[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jNcI4qHq)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=22129386)
**Weight:** 10% of module grade

**Due:** Week 05

Welcome 👋 to the repository for **IPA 1**, *CSD3121 – Developing Immersive Applications (DIA)*. For some of you, this module was previously known as *CSD3120 Introduction to VR*.

If you are seeing this repository, it means that you have **successfully joined the GitHub Classroom assignment** for this module. Congratulations! 🎉

This is an **individual programming assignment (IPA)**. For IPA 1, it will be mainly assessed **automatically via GitHub Actions**.

---

## Important: GitHub Organization Invitation (`sit-dia`)

Unfortunately, GitHub **cannot automatically invite/add you** to the `sit-dia` organization.
- We will run a script to retrieve GitHub usernames from accepted assignments
- You will soon receive an **email invitation** to join the organization
- Please **check your email (and spam folder)** carefully

The `sit-dia` organization is important because:
- All your IPA and TP repositories for this module live there
- We use its **Discussion Board** for all module communication

⚠️ **If you cannot access the discussion board, it means you have not accepted the organization invitation.**

If you are still having issues after the first lab, please contact the module the TAs (CC both so that whoever is available earlier will be able to respond):
- Leon: leon.foo@singaporetech.edu.sg
- Adi: aditya.singhania@singaporetech.edu.sg

---

## The GitHub Classroom

The GitHub Classroom will be primarily used to:
- Distribute IPAs
- Collect submissions
- Run automated tests
- Provide feedback via CI logs

There is **no submit button** for IPAs.
Your submission is simply your **latest commit pushed before the deadline**.

---

## The `sit-dia` GitHub Organization

The `sit-dia` organization houses:
- All IPA repositories for this module
- A shared **Discussion Board**

### Discussion Board

Every week, there will be a discussion post outlining:
- Weekly learning objectives
- What to focus on
- Where learning materials can be found

For reference, here is the **Week 1 post**:
[# WEEK01. Introduction · sit-dia · Discussion #3](https://github.com/orgs/sit-dia/discussions/3). All future posts will be under the same discussion category, i.e. [Lectorial Content & Discussions · Discussions · sit-dia](https://github.com/orgs/sit-dia/discussions/categories/-lectorial-content-discussions).

Again, **if you cannot access this page, you have not accepted the organization invitation**.
---

## What You Are Building in IPA 1

In **IPA 1 (2026)**, you will build a **minimal WebXR-ready Babylon.js application**.
The goal is **not** to make you an expert in Babylon.js, but to use it as an **accessible vehicle** to understand immersive application development. This will also set the foundation for IPA2 where you will contribute to an open-source immersive learning tool.

By completing IPA 1, you should be able to:
- Set up and run a TypeScript + Babylon.js project
- Understand the engine–scene lifecycle in Babylon.js
- Create a basic WebXR immersive VR experience

---

## Getting Started

If you are familiar with both GitHub and Node.js development, you can skip to the [Tasks to complete](#tasks-to-complete-in-this-lab).

### Introduction to GitHub

To complete the tasks in the IPA you will need to know your way around GitHub.

**So why are we using GitHub?**
GitHub is a tool that allows us to collaborate on code. It is a tool that is used by many software engineers and is a great tool to learn. It is also a great tool to use for our class because it allows us to easily share code and collaborate on code.

Now, if you're new to GitHub, or, have not used it extensively yet, you may want to read the basics of GitHub [here](https://docs.github.com/en/get-started/start-your-journey/about-github-and-git), or, go one step further and follow [this](https://docs.github.com/en/get-started/start-your-journey/hello-world) short tutorial. If you prefer watching a video, we found [this](https://www.youtube.com/watch?v=8Dd7KRpKeaE) video to be a good introduction to git, GitHub, and GitHub Desktop.

### The JavaScript-based Development Environment

Next, you will need to set up a JavaScript-based development environment on your computer, using the Node.js runtime. The goal is not to make you an expert in JavaScript or Node.js, but to get you familiar with the vehicle we will be using to apply immersive application development concepts. The bonus is that you will also pick up some industry-relevant interactive web-based development skills along the way.

#### Install dependencies

Make sure you have **Node.js** installed (https://nodejs.org).

As much as possible, _perform all development steps on the command line_. This will help form important associations with the underlying tools that are used in the background behind the GUI.

Then install the project dependencies:
```bash
npm install
```

#### Run the Development Server

Start the local development server using:

```bash
npm run dev
```

After running this command:
- A local URL (e.g., http://localhost:5173) will be shown in the terminal
- Open this URL in your web browser
- You should see an empty Babylon.js scene. The scene may be very minimal and WebXR functionality may not yet be fully enabled. This is expected and will be addressed as you progress through the tasks

If the page does not load or you encounter errors:
- Check the browser console for error messages
- Ensure you have run npm install successfully
- Fix any TypeScript or build errors before proceeding

#### Running Tests

Goes without saying that testing is important. We will use the GitHub actions CI system to run test files that will check whether your code prints the right message in the browser console. The test files are already provided in the repository. You do not need to modify it. This is mainly as an example of how you may implement your own automated tests.

Read the test files to understand what they are doing. Again feel free to use AI tools as necessary.

The test files will be automatically triggered on each commit but you should make sure you know how to run the test file in your local programming environment.

Locally on your own machine, you can run all tests in one go using:

```bash
npm run test
```

To run a specific test file you can, for example, use this npx to run the test file for Task 6:

```bash
npx vitest --reporter=verbose --run -t "t06"
```

Before editing the code, it is expected that all tests fail initially. Your goal is simply - make all tests pass. You should already be familiar with this TDD style of programming in other modules.
- Read the test files
- Understand what each test checks
- Modify your code until the tests pass

Note that you can see that the test files can be easily modifiable, and you can certainly hack them to your advantage. However, we can just as easily check whether you have modified the test files. So, it is not recommended to do so. You can certainly write your own test files to test your code, but the test files provided are the ones that will be used to assess your code.

## Submitting the assignment

⚠️ **Please read all submission requirements below carefully. Failure to meet said requirements may result in loss of marks, even if your code passes all the automated tests.**

There is **no submit button** for these assignments. Instead, submissions are actually **just a commit to the repository**. To submit this assignment, you need to add, commit, and push your changes to the repository. You can do this in the GitHub codespace or on your own machine.
- once your changes are pushed to the repository, the tests will automatically run
- you can check the status of the test file by going to the "Actions" tab and clicking on the latest workflow run
- you can determine whether the test passed or failed by looking at the green tick or red cross on the commit hash
- do not assume that the test files you can see are the only test files that will be run. We can simply pull all student repositories and run additional test files.

### Commit Requirement:

Your repository must contain **at least 7 commits, one for each task** (Task 1 through Task 7). This demonstrates incremental progress and helps track your development process. Each commit should have a meaningful commit message that describes what was accomplished (e.g., "Complete Task 2: Hello World", "Add VideoDome for Task 4", etc.).

### Screen Capture Requirement:

For each task from Task 2 through Task 7, you must provide a screenshot or screen recording that demonstrates the functionality you have implemented. This is to ensure that your implementation works in a real browser environment, not just in automated tests. You must share these images/GIFs/videos **as a reply to pull request #1** (the pull request automatically created by the `github-classroom` bot for this repository, available under the `Pull Requests` tab).

[Here](https://www.youtube.com/watch?v=g45OJn3UyCU) is a behind the scenes look at how we'll review your code and why including this screen captures as a reply to PR #1 will be helpful for us to grade your work smoothly. Sending code changes via pull requests is common practice in industry, and you will often get feedback from your colleagues on your code changes via feedback on pull requests. You can get an inside look at how senior developers may review pull requests by watching [this](https://www.youtube.com/watch?v=LheeJPkdCu8) video.

Your reply (or replies, no preference) should thus contain 6 screen captures, one for each of Tasks 2 through 7, either as images or videos/GIFs lasting a few seconds each. These should show the following:
- Your application running in a browser
- Key features working (e.g., scene rendering, interactions, UI updates, console messages)
- For WebXR functionality (e.g., teleportation in Task 5), you may either show it working in a real HMD screen capture, or, if you do not have access to VR hardware, show it using the [Immersive Web Emulator](https://chromewebstore.google.com/detail/immersive-web-emulator/cgffilbpcibhmcfbgggfhfolhkfbhmik?hl=en) created by [Meta](https://developers.meta.com/horizon/blog/webxr-development-immersive-web-emulator/).

In your reply (or replies) to PR #1, briefly describe what we should look out for (for example, "Tasks 2–3 completed, XR helper initialised, dragon interaction working").

### Author Declaration Requirement:

It is also important to include this **Author Declaration** in all files that you create or substantially modify for this IPA:

For `.ts` / `.js` files, you can include it within the file header comment block:
```ts
/**
 * <other file documentation...>
 *
 * Author: <Full Name> (<Student ID>)
 *
 * <list resources used, e.g.,
 *   - GitHub Copilot: code completion and to check syntax...
 *   - Claude Code: generate draft code snippets and explore
 *     alternative approaches, which were reviewed and
 *     modified before integration.
 * >
 *
 * Declaration:
 * I affirm that this file represents my own work and understanding.
 * Except where explicitly acknowledged, it has not been copied from
 * other students or external repositories. Any use of external
 * resources, tools, or assistance (including AI-based tools) has been
 * appropriately referenced.
 *
 * When using AI tools, I affirm that I have not delegated the
 * completion of this assignment to an autonomous or agentic coding
 * system in a manner where I did not meaningfully read, reason about,
 * or engage with the source code being produced (for example, by
 * submitting code generated end-to-end by an AI agent
 * without inspection, understanding, or modification).
 *
 * I take full responsibility for the integrity of this work under the
 * Singapore Institute of Technology Code of Conduct for Learners:
 * https://www.singaporetech.edu.sg/sitlearn/sites/sitlearn/files/2024-12/Code%20of%20Conduct%20for%20Learners.pdf
 */
```

For `.md` files, you can include it as a section at the end of the file:
```md
## Resources Used:
<list resources used, e.g.,
- GitHub Copilot: code completion and to check syntax...
- Claude Code: generate draft code snippets and explore
  alternative approaches, which were reviewed and
  modified before integration.
>

## Author Declaration
I affirm that this file represents my own work and understanding.
Except where explicitly acknowledged, it has not been copied from
other students or external repositories. Any use of external
resources, tools, or assistance (including AI-based tools) has been
appropriately referenced.

When using AI tools, I affirm that I have not delegated the
completion of this assignment to an autonomous or agentic coding
system in a manner where I did not meaningfully read, reason about,
or engage with the source code being produced (for example, by
submitting code generated end-to-end by an AI agent
without inspection, understanding, or modification).

I take full responsibility for the integrity of this work under the
Singapore Institute of Technology Code of Conduct for Learners:
https://www.singaporetech.edu.sg/sitlearn/sites/sitlearn/files/2024-12/Code%20of%20Conduct%20for%20Learners.pdf
```

### Assessment
- **Functionality (80%)** – Based on test cases passed for the first 7 tasks.
- **Git Workflow (20%)** – Based on whether you have made at least 7 meaningful commits, one for each task from Task 1 through Task 7.
- **No Screenshots/Videos/GIFs posted (–30%)** – Applied if there is incomplete evidence of task completion in the form of screen captures / recordings included as a reply (or replies) to PR #1.
- **Code Quality (-20%)** – Based on clarity, structure, and commenting (negative scoring applied).

## Tasks to complete in this lab

### Task 0: GitHub setup

First, make sure your **GitHub profile contains your full name**. This is important for us to be able to identify you in assessments, discussions, etc.

Second, Make sure you can successfully **clone this repository to your computer locally**. If you choose to use the GitHub codespace cloud IDE, theoretically you don't need to clone the repository, but for this exercise, you should still clone the repository to your computer locally. It is important to be able to know how to work on your project independent of cloud resources.

This task is not graded via tests, but failure to complete it will block all subsequent work.

### Task 1: Writing documentation with markdown

In the IPA and TP, you will have to write a lot of documentation. We will use markdown for all docs to facilitate version control and so that team member contributions can be easily tracked. Markdown is a lightweight markup language that is easy to learn. It is also the language that is used to write this README file, issue descriptions, discussion posts, wiki, etc.

**Create a new markdown file in this repository named `about-me.md`**. In this file, write a short paragraph about yourself. You can write about your hobbies, interests, or anything else you want to share. Also let us know what you hope to get out of this module.
- have at least two headings with different subheadings
- include at least an image in there
- You can learn about markdown [here](https://guides.github.com/features/mastering-markdown/).

This task ensures you are comfortable with documentation workflows used throughout the module.

#### Test files
The test files that check whether you have completed this task are:
- `task1-docs.test.ts`

As mentioned, you should read the test file to understand what you need to do. Reading the test file carefully will become increasingly important as the tasks get more complex.

### Task 2: Hello World

Much of the module's implementation examples and assessment will be done through Typescript. In case you have not used it before, we should certainly go through the rite of passage Hello World assignment! as the first coding task.

In particular, you need to print the "Hello, immersive world!" message in the browser console. You should write it in Typescript as we will be using Typescript for the rest of the module.  For those with other programming experience, [Learn X in Y Minutes](https://learnxinyminutes.com/) is a good resource to quickly get up to speed on the syntax of almost any language you might ever encounter. You can find their typescript page [here](https://learnxinyminutes.com/typescript/), but, certainly, do your own research as necessary to get up to speed!

Since this is the "immersive" version of Hello World, the tests will also check whether the web page rendered in the browser is actually an immersive 3D XR-ready scene. We are using BabylonJS as the graphics/XR library to do this.

Feel free to use ChatGPT, Gemini, Claude, and the likes of course, as this is not a module focused on programming fundamentals. Again, coding is just a vehicle to learn immersive application development concepts. However, as with general good practice, make sure you understand the code that is generated for you.

#### Test files
- `task2-hello.test.ts`

### Task 3: Setting up a basic Babylon.js scene

Extend your application so that it contains a **properly constructed Babylon.js scene** following standard setup patterns. This task establishes the core scene structure that subsequent tasks will build upon.

As part of this task, you should aim to implement a minimal but meaningful scene, for example:
- Create at least two basic scene primitives (e.g., a sphere and a plane or ground)
- Add a camera that allows the scene to be viewed clearly
- Add at least one light source so objects are visible
- Attach a simple GUI element (e.g., a text label) to the scene to demonstrate basic UI setup

The goal is to ensure that your scene is structurally sound and ready to support later features such as environments, interactions, and dynamic UI updates.

For assessment purposes, the emphasis is on **correct scene structure and initialization**, not on visual fidelity, layout aesthetics, or immersive effects. When run in a real browser environment, your scene should nevertheless render without errors and behave as expected.

As with other tasks, you are expected to inspect the test file to understand the exact requirements being validated.

#### Test files
- `task3-scene.test.ts`

### Task 4: Creating a basic immersive environment with a VideoDome

Add an **immersive background environment** using a Babylon.js `VideoDome`.

Your implementation should correctly create and attach a `VideoDome` to the scene as a skydome-style background environment using a video texture. This task introduces a common immersive media format that provides a lightweight way to represent immersive surrounding environments.
- see Babylon.js documentation for VideoDome: https://doc.babylonjs.com/features/featuresDeepDive/environment/360VideoDome
- the video used can be any equirectangular video of your choice, e.g., some free ones from Mettle: https://www.mettle.com/360vr-master-series-free-360-downloads-page/

For assessment purposes, the emphasis is on **correct structure and scene integration**, rather than actual video playback, rendering fidelity, or immersive behavior. When run in a real browser environment, your implementation should nevertheless behave as expected.

#### Test files
- `task4-env.test.ts`

### Task 5: Enabling WebXR immersive VR mode

Enable **WebXR immersive VR support** in your application using Babylon.js.

This task focuses on correctly initializing Babylon.js WebXR support using the standard helper APIs so that your scene is structurally ready for immersive VR when run in a compatible environment.
- configure the scene to support immersive VR mode
- configure standard teleportation option, e.g., set the teleportation animation time to 2s

You are **not required** to own a VR headset or use an XR-capable browser to complete this task. For assessment purposes, the emphasis is on **proper WebXR initialization**, not on actually entering an immersive VR session.

As with previous tasks, WebXR functionality is validated structurally during testing. When run in a real XR-capable browser with supported hardware, your application should be able to enter immersive VR mode as expected.

#### Test files
- `task5-xr.test.ts`

### Task 6: Creating basic click and drag interactions in WebXR

Implement **basic interaction logic** that allows users to interact with objects in the scene in both desktop and XR contexts. This task focuses on wiring scene content, interaction handlers, and application state changes together in a meaningful way.

As part of this task, you should load a custom 3D model asset into the scene to serve as an interactive reference object. For example, you may use the following dragon model:

https://raw.githubusercontent.com/BabylonJS/Assets/master/meshes/Georgia-Tech-Dragon/dragon.glb

Examples of interactions you should aim to implement include:
- Loading the dragon model into the scene as a visible object
- Enabling pick or select interactions on a scene object (e.g., clicking or selecting the sphere)
- Changing application or scene state when the interaction occurs (e.g., toggling a flag, updating object properties)
- Attaching interaction logic directly to scene objects rather than relying on hardcoded or global handlers

The goal is to demonstrate how user actions are translated into detectable state changes that can drive application behavior. Your interaction logic should be implemented in a way that would function correctly in a real browser or XR-capable environment.

For assessment purposes, interactions are validated **structurally**, not through actual pointer or controller simulation. The emphasis is on correct setup and wiring of interaction logic rather than interaction fidelity or UX polish.

#### Test files
- `task6-interact.test.ts`

### Task 7: Create a dynamic UI element to track scene state in WebXR

Extend your application by adding a **dynamic UI element** that reflects changes in scene state during execution. This task builds directly on Task 6 by surfacing interaction-driven state changes through an on-screen or in-world UI element.

As part of this task, you should build on the scene content and interactions you have already implemented. For example:
- Use the dragon model loaded in Task 6:
  https://raw.githubusercontent.com/BabylonJS/Assets/master/meshes/Georgia-Tech-Dragon/dragon.glb
- Display a text label that updates continuously to show the distance between the sphere and the dragon, formatted as "d: X.XX"
- Make the UI update automatically as the sphere is moved or manipulated
- Implement the UI as a simple in-world panel (e.g., a plane with an `AdvancedDynamicTexture` and a `TextBlock`)

The goal is to demonstrate a meaningful **scene state → UI update** loop that is common in interactive XR applications, such as distance indicators, status readouts, or contextual feedback.

For assessment purposes, the emphasis is on correctly connecting scene state changes to UI updates. Automated tests validate structure and application logic in a headless environment; actual rendering quality, layout aesthetics, or immersive fidelity are not required. However, your implementation should still behave as intended when run in a real browser environment.

#### Test files
- `task7-ui.test.ts`

## Summary

By completing these tasks, you will have built a foundational immersive application using Babylon.js and WebXR. Again, the goal is not to make you an expert in Babylon.js, but to use it as an accessible vehicle to understand immersive application development. This will also set the foundation for IPA2 where you will contribute to an open-source immersive learning tool.

Note that the individual quizzes will also test your ability to apply some of these concepts through the implementation environment you have learned in this IPA, so make sure you understand what you are doing here.
