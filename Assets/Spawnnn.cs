using Fusion;
using UnityEngine;

public class Spawnnn : NetworkBehaviour
{
    [SerializeField] private GameObject _chess;

    public override void Spawned()
    {
        base.Spawned();
        if(!Runner.IsSharedModeMasterClient) Runner.Spawn(_chess, transform.position, transform.rotation);
    }
}
