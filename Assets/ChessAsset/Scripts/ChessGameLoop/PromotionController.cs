using ChessEnums;
using Digiphy;
using Fusion;
using UnityEngine;

namespace ChessMainLoop
{
    public class PromotionController :  SingletonNetworkedReplaceable<PromotionController>
    {
        [SerializeField] private GameObject _blackPieces;
        [SerializeField] private GameObject _whitePieces;

        [SerializeField] private Queen _whiteQueen;
        [SerializeField] private Queen _blackQueen;
        [SerializeField] private Bishop _whiteBishop;
        [SerializeField] private Bishop _blackBishop;
        [SerializeField] private Rook _whiteRook;
        [SerializeField] private Rook _blackRook;
        [SerializeField] private Knight _whiteKnight;
        [SerializeField] private Knight _blackKnight;

        public void PawnPromotionMenu(SideColor color)
        {
            if (color == SideColor.White) _whitePieces.SetActive(true);
            else _blackPieces.SetActive(true);
        }

        [Rpc(sources: RpcSources.All, targets: RpcTargets.All, HostMode = RpcHostMode.SourceIsServer)]
        public void RPC_PieceSelected(ChessPieceType pieceIndex, int Row, int Column)
        {
            (int, int) pawnLocation = (Row, Column);
            _whitePieces.SetActive(false);
            _blackPieces.SetActive(false);

            switch(pieceIndex)
            {
                case ChessPieceType.BlackQueen:
                    GameManager.Instance.SelectedPromotion(Instantiate(_blackQueen), pieceIndex, pawnLocation);
                    break;
                case ChessPieceType.WhiteQueen:
                    GameManager.Instance.SelectedPromotion(Instantiate(_whiteQueen), pieceIndex, pawnLocation);
                    break;
                case ChessPieceType.BlackRook:
                    GameManager.Instance.SelectedPromotion(Instantiate(_blackRook), pieceIndex, pawnLocation);
                    break;
                case ChessPieceType.WhiteRook:
                    GameManager.Instance.SelectedPromotion(Instantiate(_whiteRook), pieceIndex, pawnLocation);
                    break;
                case ChessPieceType.BlackBishop:
                    GameManager.Instance.SelectedPromotion(Instantiate(_blackBishop), pieceIndex, pawnLocation);
                    break;
                case ChessPieceType.WhiteBishop:
                    GameManager.Instance.SelectedPromotion(Instantiate(_whiteBishop), pieceIndex, pawnLocation);
                    break;
                case ChessPieceType.BlackKnight:
                    GameManager.Instance.SelectedPromotion(Instantiate(_blackKnight), pieceIndex, pawnLocation);
                    break;
                case ChessPieceType.WhiteKnight:
                    GameManager.Instance.SelectedPromotion(Instantiate(_whiteKnight), pieceIndex, pawnLocation);
                    break;
            }
        }


    }
}