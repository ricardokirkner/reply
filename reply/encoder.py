import numpy

class Encoder(object):
    def encode(self, item):
        raise NotImplementedError()

    def decode(self, item):
        raise NotImplementedError()


class DummyEncoder(Encoder):
    def encode(self, item):
        return item
    def decode(self, item):
        return item


class SpaceEncoder(Encoder):
    def __init__(self, space):
        self.space = space

    def encode(self, item):
        """
        >>> from reply.datatypes import Integer, Space
        >>> space = Space({'choice': Integer(0, 9)})
        >>> encoder = SpaceEncoder(space)
        >>> encoder.encode({'choice': 5})
        (5,)
        """
        encoded_item = [item[key] for key in self.space.get_names_list()]
        return tuple(encoded_item)

    def decode(self, encoded_item):
        """
        >>> from reply.datatypes import Integer, Space
        >>> space = Space({'choice': Integer(0, 9)})
        >>> encoder = SpaceEncoder(space)
        >>> encoder.decode((5,))
        {'choice': 5}
        """
        item = {}
        for key, value in zip(self.space.get_names_list(), encoded_item):
            item[key] = value
        return item


class StateActionEncoder(SpaceEncoder):
    def __init__(self, state_encoder, action_encoder):
        self.encoder = {'state': state_encoder, 'action': action_encoder}

    def encode(self, item):
        """
        >>> from reply.datatypes import Integer, Space
        >>> state_space = Space({'state': Integer(0, 0)})
        >>> action_space = Space({'choice': Integer(0, 9)})
        >>> state_encoder = SpaceEncoder(state_space)
        >>> action_encoder = SpaceEncoder(action_space)
        >>> encoder = StateActionEncoder(state_encoder, action_encoder)
        >>> encoder.encode(({'state': 0}, {'choice': 3}))
        (0, 3)
        >>> encoder.encode(({'state': 0},))
        (0,)
        >>> encoder.encode({'state': 0})
        (0,)
        """
        action = None
        if isinstance(item, (list, tuple)):
            if len(item) == 2:
                state, action = item
            elif len(item) == 1:
                state = item[0]
            else:
                raise ValueError("Invalid item", item)
        else:
            state = item
        encoded_state = self.encoder['state'].encode(state)
        encoded_item = encoded_state
        if action is not None:
            encoded_action = self.encoder['action'].encode(action)
            encoded_item += encoded_action
        return encoded_item

    def decode(self, encoded_item):
        """
        >>> from reply.datatypes import Integer, Space
        >>> state_space = Space({'state': Integer(0, 0)})
        >>> action_space = Space({'choice': Integer(0, 9)})
        >>> state_encoder = SpaceEncoder(state_space)
        >>> action_encoder = SpaceEncoder(action_space)
        >>> encoder = StateActionEncoder(state_encoder, action_encoder)
        >>> encoder.decode((0, 3))
        ({'state': 0}, {'choice': 3})
        >>> encoder.decode((0,))
        ({'state': 0},)
        >>> encoder.decode(0)
        ({'state': 0},)
        """
        encoded_action = None
        if isinstance(encoded_item, (list, tuple)):
            if len(encoded_item) == 2:
                encoded_state, encoded_action = encoded_item
            elif len(encoded_item) == 1:
                encoded_state = encoded_item[0]
            else:
                raise ValueError("Invalid encoded item", encoded_item)
        else:
            encoded_state = encoded_item
        state = self.encoder['state'].decode([encoded_state])
        item = (state,)
        if encoded_action is not None:
            action = self.encoder['action'].decode([encoded_action])
            item += (action,)
        return item


if __name__ == '__main__':
    import doctest
    doctest.testmod()
