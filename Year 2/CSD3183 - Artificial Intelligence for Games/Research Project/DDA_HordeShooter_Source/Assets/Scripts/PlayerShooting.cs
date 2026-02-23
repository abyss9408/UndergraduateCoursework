// PlayerShooting.cs - Handles weapon mechanics with DDA integration
using UnityEngine;

public class PlayerShooting : MonoBehaviour
{
    [Header("Weapon Settings")]
    public float fireRate = 10f;
    public float damage = 25f;
    public float range = 100f;
    public int maxAmmo = 30;
    public float reloadTime = 2f;

    [Header("Visual Effects")]
    public GameObject muzzleFlash;
    public LineRenderer bulletTrail;
    public float trailTime = 0.1f;

    private Camera playerCamera;
    private PlayerPerformanceTracker performanceTracker;
    private float nextFireTime = 0f;
    private int currentAmmo;
    private bool isReloading = false;
    private AudioSource audioSource;

    void Start()
    {
        playerCamera = GetComponentInChildren<Camera>();
        performanceTracker = GetComponent<PlayerPerformanceTracker>();
        audioSource = GetComponent<AudioSource>();

        currentAmmo = maxAmmo;

        if (bulletTrail != null)
        {
            bulletTrail.enabled = false;
            bulletTrail.startWidth = 0.02f;
            bulletTrail.endWidth = 0.02f;
        }
    }

    void Update()
    {
        HandleShooting();
        HandleReload();
    }

    void HandleShooting()
    {
        if (isReloading) return;

        if (Input.GetButton("Fire1") && Time.time >= nextFireTime && currentAmmo > 0)
        {
            Shoot();
            nextFireTime = Time.time + 1f / fireRate;
        }
    }

    void HandleReload()
    {
        if (Input.GetKeyDown(KeyCode.R) && currentAmmo < maxAmmo && !isReloading)
        {
            StartCoroutine(Reload());
        }

        // Auto reload when empty
        if (currentAmmo <= 0 && !isReloading)
        {
            StartCoroutine(Reload());
        }
    }

    void Shoot()
    {
        currentAmmo--;
        performanceTracker?.OnShotFired();

        // Raycast from camera center
        Ray ray = playerCamera.ScreenPointToRay(new Vector3(Screen.width / 2, Screen.height / 2, 0));
        RaycastHit hit;

        Vector3 targetPoint;
        bool hitSomething = false;

        if (Physics.Raycast(ray, out hit, range))
        {
            targetPoint = hit.point;
            hitSomething = true;

            // Check if we hit an enemy
            EnemyAI enemy = hit.collider.GetComponent<EnemyAI>();
            if (enemy != null)
            {
                enemy.TakeDamage(damage);
                performanceTracker?.OnShotHit();
            }
        }
        else
        {
            targetPoint = ray.origin + ray.direction * range;
        }

        // Visual effects
        ShowMuzzleFlash();
        ShowBulletTrail(ray.origin, targetPoint);

        // Audio effect (if you have an audio source)
        if (audioSource != null)
        {
            // You can add a gunshot sound clip here
        }

        Debug.Log($"Shot fired! Ammo: {currentAmmo}/{maxAmmo} | Hit: {hitSomething}");
    }

    void ShowMuzzleFlash()
    {
        if (muzzleFlash != null)
        {
            muzzleFlash.SetActive(true);
            Invoke(nameof(HideMuzzleFlash), 0.05f);
        }
    }

    void HideMuzzleFlash()
    {
        if (muzzleFlash != null)
            muzzleFlash.SetActive(false);
    }

    void ShowBulletTrail(Vector3 start, Vector3 end)
    {
        if (bulletTrail != null)
        {
            StartCoroutine(DrawTrail(start, end));
        }
    }

    System.Collections.IEnumerator DrawTrail(Vector3 start, Vector3 end)
    {
        bulletTrail.enabled = true;
        bulletTrail.SetPosition(0, start);
        bulletTrail.SetPosition(1, end);

        yield return new WaitForSeconds(trailTime);

        bulletTrail.enabled = false;
    }

    System.Collections.IEnumerator Reload()
    {
        isReloading = true;
        Debug.Log("Reloading...");

        yield return new WaitForSeconds(reloadTime);

        currentAmmo = maxAmmo;
        isReloading = false;
        Debug.Log("Reload complete!");
    }

    public int GetCurrentAmmo() => currentAmmo;
    public int GetMaxAmmo() => maxAmmo;
    public bool IsReloading() => isReloading;
}