export class SnapshotEpoch {
  private value = 0;

  beginHttpRequest(): number {
    return this.value;
  }

  acceptHttpResponse(requestEpoch: number): boolean {
    if (requestEpoch !== this.value) return false;
    this.value += 1;
    return true;
  }

  isCurrent(requestEpoch: number): boolean {
    return requestEpoch === this.value;
  }

  acceptStreamSnapshot(): void {
    this.value += 1;
  }

  acceptAuthoritativeSnapshot(): void {
    this.value += 1;
  }
}
