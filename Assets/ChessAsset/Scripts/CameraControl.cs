using UnityEngine;

public class CameraControl : MonoBehaviour
{
    [SerializeField]
    private float _movementSpeed = 10f;
    [SerializeField]
    private float _fastMovementSpeed = 100f;
    [SerializeField]
    private float _freeLookSensitivity = 3f;
    [SerializeField]
    private float _zoomSensitivity = 10f;
    [SerializeField]
    private float _fastZoomSensitivity = 50f;
    [SerializeField]
    private float _panSensitivity = 0.3f;
    private Rigidbody _rigidbody;

    private void Start()
    {
        _rigidbody = GetComponent<Rigidbody>();
    }

    void Update()
    {
        _rigidbody.velocity = Vector3.zero;
        _rigidbody.angularVelocity = Vector3.zero;

        var _fastMode = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
        var _movementSpeed = _fastMode ? _fastMovementSpeed : this._movementSpeed;

        if ((Input.GetKey(KeyCode.A)))
        {
            transform.position = TryToMove(transform.position, (-transform.right * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.Space))
        {
            transform.position = TryToMove(transform.position, (Vector3.up * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.D))
        {
            transform.position = TryToMove(transform.position, (transform.right * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.W))
        {
            transform.position = TryToMove(transform.position, (transform.forward * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.S))
        {
            transform.position = TryToMove(transform.position, (-transform.forward * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.Q))
        {
            transform.position = TryToMove(transform.position, (transform.up * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.E))
        {
            transform.position = TryToMove(transform.position, (-transform.up * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.R) || Input.GetKey(KeyCode.PageUp))
        {
            transform.position = TryToMove(transform.position, (Vector3.up * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.F) || Input.GetKey(KeyCode.PageDown))
        {
            transform.position = TryToMove(transform.position, (-Vector3.up * _movementSpeed * Time.deltaTime));
        }

        if (Input.GetKey(KeyCode.Mouse1))
        {
            float newRotationX = transform.localEulerAngles.y + Input.GetAxis("Mouse X") * _freeLookSensitivity;
            float newRotationY = transform.localEulerAngles.x - Input.GetAxis("Mouse Y") * _freeLookSensitivity;
            transform.localEulerAngles = new Vector3(newRotationY, newRotationX, 0f);
        }

        if (Input.GetKey(KeyCode.Mouse2))
        {
            transform.position = TryToMove(transform.position, (- transform.right * Input.GetAxis("Mouse X") * _panSensitivity));
            transform.position = TryToMove(transform.position, (- transform.up * Input.GetAxis("Mouse Y") * _panSensitivity));
        }

        float axis = Input.GetAxis("Mouse ScrollWheel");
        if (axis != 0)
        {
            var zoomSensitivity = _fastMode ? this._fastZoomSensitivity : this._zoomSensitivity;
            transform.position = TryToMove(transform.position, transform.forward * axis * zoomSensitivity);
        }       

    }

    /// <summary>
    /// Checks if the original vector is within parameters after offset translation.
    /// </summary>
    /// <returns>Translated vector if its within parameters, or original vector if not</returns>
    private Vector3 TryToMove(Vector3 _position, Vector3 _offset)
    {
        if ((_position + _offset).x < 35f && (_position + _offset).x > -35f)
        {
            _position.x += _offset.x;
        }

        if ((_position + _offset).z < 35f && (_position + _offset).z > -35)
        {
            _position.z += _offset.z;
        }

        if ((_position + _offset).y < 35f && (_position + _offset).y > 2f)
        {
            _position.y += _offset.y;
        }

        return _position;
    }

}
