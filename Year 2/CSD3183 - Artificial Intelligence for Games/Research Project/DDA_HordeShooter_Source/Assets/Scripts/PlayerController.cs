// PlayerController.cs - More robust rotation system with drift prevention
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float walkSpeed = 5f;
    public float runSpeed = 8f;
    public float jumpForce = 5f;
    public float mouseSensitivity = 2f;

    [Header("Camera Settings")]
    public float maxLookAngle = 80f;
    public float mouseDeadzone = 0.02f; // Increased deadzone
    public bool invertY = false;

    [Header("Rotation Protection")]
    public bool enableRotationProtection = true;
    public float maxAllowedRotationPerFrame = 10f; // Max degrees per frame
    public float inputRequiredThreshold = 0.01f; // Minimum input to allow rotation

    private Rigidbody rb;
    private Camera playerCamera;
    private float verticalRotation = 0;
    private bool isGrounded;
    private bool isRunning;
    private bool cursorLocked = true;

    // Rotation tracking and protection
    private Vector3 lastFrameRotation;
    private Vector3 lockedRotation;
    private bool rotationLocked = false;
    private float noInputTimer = 0f;
    private float lockRotationAfterSeconds = 0.1f; // Lock rotation after no input for this long

    // Ground detection
    [Header("Ground Detection")]
    public LayerMask groundMask = 1;
    public float groundCheckDistance = 0.1f;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        playerCamera = GetComponentInChildren<Camera>();

        // Freeze rigidbody rotation to prevent physics interference
        if (rb != null)
        {
            rb.freezeRotation = true;
        }

        SetCursorLock(true);

        if (playerCamera == null)
        {
            Debug.LogError("No camera found in player children!");
            return;
        }

        // Initialize rotation values
        lastFrameRotation = transform.eulerAngles;
        lockedRotation = transform.eulerAngles;

        // Get initial camera rotation
        Vector3 cameraEulers = playerCamera.transform.localEulerAngles;
        verticalRotation = cameraEulers.x;
        if (verticalRotation > 180f) verticalRotation -= 360f;
    }

    void Update()
    {
        HandleInput();
        HandleMouseLook();
        CheckGrounded();

        if (enableRotationProtection)
        {
            EnforceRotationProtection();
        }
    }

    void FixedUpdate()
    {
        HandleMovement();
    }

    void HandleInput()
    {
        isRunning = Input.GetKey(KeyCode.LeftShift);

        if (Input.GetKeyDown(KeyCode.Space) && isGrounded)
        {
            Jump();
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            SetCursorLock(!cursorLocked);
        }

        // Reset rotation on R key (emergency fix)
        if (Input.GetKeyDown(KeyCode.R) && Input.GetKey(KeyCode.LeftControl))
        {
            EmergencyRotationReset();
        }
    }

    void SetCursorLock(bool locked)
    {
        cursorLocked = locked;
        if (locked)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
        else
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
    }

    void HandleMovement()
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        Vector3 direction = transform.right * horizontal + transform.forward * vertical;
        direction.Normalize();

        float currentSpeed = isRunning ? runSpeed : walkSpeed;
        Vector3 movement = direction * currentSpeed;

        movement.y = rb.linearVelocity.y;
        rb.linearVelocity = movement;
    }

    void HandleMouseLook()
    {
        if (!cursorLocked)
        {
            noInputTimer += Time.deltaTime;
            return;
        }

        // Get raw mouse input
        float rawMouseX = Input.GetAxis("Mouse X");
        float rawMouseY = Input.GetAxis("Mouse Y");

        // Apply deadzone - if input is too small, treat as zero
        float mouseX = (Mathf.Abs(rawMouseX) > mouseDeadzone) ? rawMouseX : 0f;
        float mouseY = (Mathf.Abs(rawMouseY) > mouseDeadzone) ? rawMouseY : 0f;

        // Check if we have meaningful input
        bool hasInput = (Mathf.Abs(mouseX) > inputRequiredThreshold || Mathf.Abs(mouseY) > inputRequiredThreshold);

        if (hasInput)
        {
            noInputTimer = 0f;
            rotationLocked = false;

            // Limit rotation speed per frame
            mouseX = Mathf.Clamp(mouseX, -maxAllowedRotationPerFrame / mouseSensitivity,
                                maxAllowedRotationPerFrame / mouseSensitivity);
            mouseY = Mathf.Clamp(mouseY, -maxAllowedRotationPerFrame / mouseSensitivity,
                                maxAllowedRotationPerFrame / mouseSensitivity);

            // Apply sensitivity
            mouseX *= mouseSensitivity;
            mouseY *= mouseSensitivity * (invertY ? 1f : -1f);

            // Apply horizontal rotation (Y-axis)
            float newYRotation = transform.eulerAngles.y + mouseX;
            transform.rotation = Quaternion.Euler(0, newYRotation, 0);

            // Apply vertical rotation to camera
            verticalRotation += mouseY;
            verticalRotation = Mathf.Clamp(verticalRotation, -maxLookAngle, maxLookAngle);
            playerCamera.transform.localRotation = Quaternion.Euler(verticalRotation, 0f, 0f);

            // Update locked rotation
            lockedRotation = transform.eulerAngles;
        }
        else
        {
            noInputTimer += Time.deltaTime;

            // Lock rotation if no input for too long
            if (noInputTimer > lockRotationAfterSeconds && !rotationLocked)
            {
                rotationLocked = true;
                lockedRotation = transform.eulerAngles;
            }
        }
    }

    void EnforceRotationProtection()
    {
        Vector3 currentRotation = transform.eulerAngles;

        // If rotation is locked, enforce it
        if (rotationLocked)
        {
            if (Vector3.Distance(currentRotation, lockedRotation) > 0.1f)
            {
                Debug.LogWarning($"Enforcing locked rotation. Was: {currentRotation}, Locked: {lockedRotation}");
                transform.rotation = Quaternion.Euler(lockedRotation);
                return;
            }
        }

        // Check for unexpected rotation changes
        Vector3 rotationDelta = currentRotation - lastFrameRotation;

        // Normalize angle differences
        for (int i = 0; i < 3; i++)
        {
            if (rotationDelta[i] > 180f) rotationDelta[i] -= 360f;
            if (rotationDelta[i] < -180f) rotationDelta[i] += 360f;
        }

        // Check if rotation change is too large or unexpected
        bool unexpectedRotation = (Mathf.Abs(rotationDelta.x) > maxAllowedRotationPerFrame ||
                                  Mathf.Abs(rotationDelta.z) > maxAllowedRotationPerFrame) &&
                                  noInputTimer > 0.05f; // No input for 50ms

        if (unexpectedRotation)
        {
            Debug.LogWarning($"Prevented unexpected rotation: {rotationDelta}, reverting to: {lastFrameRotation}");

            // Revert to last good rotation
            Vector3 correctedRotation = new Vector3(0, lastFrameRotation.y, 0);
            transform.rotation = Quaternion.Euler(correctedRotation);

            // Also fix camera if it's wrong
            if (Mathf.Abs(rotationDelta.x) > 1f)
            {
                playerCamera.transform.localRotation = Quaternion.Euler(verticalRotation, 0f, 0f);
            }
        }
        else
        {
            // Update last frame rotation only if the change was acceptable
            lastFrameRotation = transform.eulerAngles;
        }
    }

    void CheckGrounded()
    {
        isGrounded = Physics.Raycast(transform.position, Vector3.down, 1.1f, groundMask);
    }

    void Jump()
    {
        rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
    }

    public bool IsMoving()
    {
        return rb.linearVelocity.magnitude > 0.1f;
    }

    public bool IsRunning()
    {
        return isRunning && IsMoving();
    }

    // Emergency methods
    void EmergencyRotationReset()
    {
        Debug.Log("Emergency rotation reset activated!");

        // Reset to identity
        transform.rotation = Quaternion.identity;
        verticalRotation = 0;
        playerCamera.transform.localRotation = Quaternion.identity;

        // Reset tracking variables
        lastFrameRotation = Vector3.zero;
        lockedRotation = Vector3.zero;
        rotationLocked = false;
        noInputTimer = 0f;
    }

    [ContextMenu("Reset Rotation")]
    public void ResetRotation()
    {
        EmergencyRotationReset();
    }

    [ContextMenu("Toggle Rotation Protection")]
    public void ToggleRotationProtection()
    {
        enableRotationProtection = !enableRotationProtection;
        Debug.Log($"Rotation protection: {(enableRotationProtection ? "ENABLED" : "DISABLED")}");
    }

    // Debug method to check what might be causing rotation
    void OnDrawGizmos()
    {
        if (Application.isPlaying && enableRotationProtection)
        {
            // Draw forward direction
            Gizmos.color = Color.blue;
            Gizmos.DrawRay(transform.position, transform.forward * 2f);

            // Draw locked direction if locked
            if (rotationLocked)
            {
                Gizmos.color = Color.red;
                Vector3 lockedForward = Quaternion.Euler(lockedRotation) * Vector3.forward;
                Gizmos.DrawRay(transform.position + Vector3.up * 0.5f, lockedForward * 1.5f);
            }
        }
    }
}