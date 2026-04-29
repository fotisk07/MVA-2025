import torch
import torch_geometric


def image_to_graph(image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None) -> torch_geometric.data.Data:
    """Converts an image tensor to a PyTorch Geometric Data object.
    COMPLETE

    Arguments:
    ---------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    -------
    torch_geometric.data.Data
        Graph representation of the image.

    """
    # Assumptions (remove it for the bonus)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 2, "Expected padding of 2 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 5, "Expected kernel size of 5x5."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    C, H, W = image.shape
    device = image.device

    # node features: (H*W, C)
    x = image.permute(1, 2, 0).reshape(-1, C)

    edge_index = []
    edge_attr = []

    # kernel parameters (fixed by assumptions)
    k = 2  # radius for 5x5

    def node_id(h, w):
        return h * W + w

    for h in range(H):
        for w in range(W):
            i = node_id(h, w)
            for dh in range(-k, k + 1):
                for dw in range(-k, k + 1):
                    h2, w2 = h + dh, w + dw
                    if 0 <= h2 < H and 0 <= w2 < W:
                        j = node_id(h2, w2)
                        edge_index.append([j, i])  # message j -> i
                        edge_attr.append([dh, dw])

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=device)

    return torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def graph_to_image(
    data: torch.Tensor,
    height: int,
    width: int,
    conv2d: torch.nn.Conv2d | None = None,
) -> torch.Tensor:
    """Converts a graph representation of an image to an image tensor.

    Arguments:
    ---------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    -------
    torch.Tensor
        Image tensor of shape (C, H, W).

    """
    # Assumptions (remove it for the bonus)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 2, "Expected padding of 2 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 5, "Expected kernel size of 5x5."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    N, C = data.shape
    image = data.reshape(height, width, C).permute(2, 0, 1)
    return image


class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """A Message Passing layer that simulates a given Conv2d layer."""

    def __init__(self, conv2d: torch.nn.Conv2d):
        # <TO IMPLEMENT>
        # Don't forget to call the parent constructor with the correct aguments
        # super().__init__(<arguments>)
        # </TO IMPLEMENT>
        super().__init__(aggr="add")
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size

        # conv2d.weight has shape (C_out, C_in, 5, 5)
        # we reshape it to index by (Δh, Δw)
        weight = conv2d.weight.detach()
        self.weight = torch.nn.Parameter(weight)

        # center index for offsets
        self.k = self.kernel_size[0] // 2

    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ---------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        -------
        torch.Tensor
            The message to be passed for each edge (of size COMPLETE)

        """
        dh = edge_attr[:, 0].long() + self.k
        dw = edge_attr[:, 1].long() + self.k
        W = self.weight[:, :, dh, dw].permute(2, 0, 1)
        return torch.einsum("eoc,ec->eo", W, x_j)
