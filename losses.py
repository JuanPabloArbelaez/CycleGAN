import torch



def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    adversarial_loss_AB, fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    adversarial_loss_BA, fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    adversarial_loss = adversarial_loss_AB + adversarial_loss_BA

    identity_loss_AB = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_BA = get_identity_loss(real_B, gen_AB, identity_criterion)
    identity_loss = identity_loss_AB + identity_loss_BA

    cycle_consistency_loss_AB = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_consistency_loss_BA = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    cycle_consistency_loss = cycle_consistency_loss_AB + cycle_consistency_loss_BA

    gen_loss = adversarial_loss + identity_loss + cycle_consistency_loss

    return gen_loss, fake_A, fake_B
    

def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    disc_real_pred = disc_X(real_X)
    disc_fake_pred = disc_X(fake_X)
    disc_real_loss = adv_criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_fake_loss = adv_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_loss = (disc_real_loss + disc_fake_loss) / 2
    
    return disc_loss


def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    fake_Y = gen_XY(real_X)
    disc_pred = disc_Y(fake_Y.detach())
    adversarial_loss = adv_criterion(disc_pred, torch.ones_like(disc_pred))

    return adversarial_loss, fake_Y


def get_identity_loss(real_X, gen_YX, identity_criterion):
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(real_X, identity_X)

    return identity_loss, identity_X


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(real_X, cycle_X)

    return cycle_loss, cycle_X
